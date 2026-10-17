"""Same-engine, fixed-input investigation of pretrained NVFP4 CPU residency."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import torch
import nvfp4_real_model as common

ROOT = Path(__file__).resolve().parent


def fingerprint(tensor):
    data = tensor.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def delta(a, b):
    assert a.shape == b.shape
    d = (a.float() - b.float()).abs()
    return {"equal": torch.equal(a, b), "max_abs": d.max().item(),
            "mean_abs": d.mean().item(), "argmax_equal": bool(torch.equal(a.argmax(-1), b.argmax(-1))),
            "changed": int(torch.count_nonzero(d))}


def trace_first_layer(model, llm, prompts, tag):
    from vllm import SamplingParams
    records = {}
    handles = []
    modules = dict(model.named_modules())

    def tensors(value):
        if isinstance(value, torch.Tensor):
            return [value.detach().clone()]
        if isinstance(value, (list, tuple)):
            return [t for v in value for t in tensors(v)]
        if isinstance(value, dict):
            return [t for v in value.values() for t in tensors(v)]
        return []

    def hook(name):
        def collect(module, inputs, kwargs, output):
            records.setdefault(name, []).append((tensors((inputs, kwargs)), tensors(output)))
        return collect

    for name, module in modules.items():
        if name.startswith("model."):
            handles.append(module.register_forward_hook(hook(name), with_kwargs=True))
    raw = []
    original_logits = model.compute_logits
    def capture_logits(*a, **kw):
        out = original_logits(*a, **kw)
        raw.append(out.clone())
        return out
    model.compute_logits = capture_logits
    for _ in range(2):
        llm.generate(prompts[:1], SamplingParams(temperature=0, max_tokens=1, ignore_eos=True), use_tqdm=False)
    for h in handles:
        h.remove()
    model.compute_logits = original_logits
    report = {}
    for name, values in records.items():
        if len(values) != 2:
            continue
        result = {}
        for side, index in [("input", 0), ("output", 1)]:
            result[side] = [delta(a, b) for a, b in zip(values[0][index], values[1][index], strict=True)]
        report[name] = result
        print("TRACE", name, json.dumps(result), flush=True)
    # Replay real first-layer dense inputs with no scheduler, attention or MoE.
    direct = {}
    with torch.inference_mode():
        for name, values in records.items():
            layer = modules[name]
            if "qkv_proj" not in name or not hasattr(layer, "quant_method") or not values[0][0]:
                continue
            x = values[0][0][0]
            outputs = [layer.quant_method.apply(layer, x).clone() for _ in range(5)]
            direct[name] = {"input_shape": list(x.shape), "method": type(layer.quant_method).__name__,
                            "repeat_deltas": [delta(outputs[0], o) for o in outputs[1:]]}
            print("DIRECT", name, json.dumps(direct[name]), flush=True)
            break
    (ROOT / f"nvfp4-trace-{tag}.json").write_text(json.dumps({"trace": report, "direct_dense": direct,
        "logits_delta": delta(raw[0], raw[1])}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=["baseline", "patched"])
    ap.add_argument("--tag", default="same-process")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--stable-alignment", action="store_true")
    ap.add_argument("--skip-hashes", action="store_true")
    args = ap.parse_args()
    scope = vars(common).copy()
    if os.environ.get("PR5090_FULL_MAIN") != "1":
        common.CompressedTensorsW4A4Nvfp4MoEMethod.get_post_load_parameter_name = common.definition(
            ROOT / "sources/nvfp4_method.py", "get_post_load_parameter_name", scope,
            "CompressedTensorsW4A4Nvfp4MoEMethod")
        common.loader_utils.device_loading_context = common.definition(
            ROOT / "sources" / f"{args.arm}_loader_utils.py", "device_loading_context", scope)
    from vllm import LLM, SamplingParams
    llm = LLM(model=str(ROOT / "models/Qwen3-30B-A3B-NVFP4"), dtype="bfloat16",
              max_model_len=1024, max_num_seqs=4, max_num_batched_tokens=1024,
              enforce_eager=True, compilation_config=0, gpu_memory_utilization=0.80,
              kv_cache_memory_bytes=512 * 1024**2, enable_prefix_caching=False,
              attention_backend="TRITON_ATTN", moe_backend="marlin", cpu_offload_gb=1,
              cpu_offload_params={"experts"}, seed=0,
              **({"linear_backend": "cutlass"} if os.environ.get("PR5090_FULL_MAIN") == "1" else {}))
    core = llm.llm_engine.engine_core.engine_core
    runner = core.model_executor.driver_worker.worker.model_runner
    model = runner.get_model()
    params = {n: p for n, p in model.named_parameters()}
    hashes = {} if args.trace or args.skip_hashes else {n: {"shape": list(p.shape), "dtype": str(p.dtype), "sha256": fingerprint(p),
                  "uva": bool(getattr(p, "_vllm_is_uva_offloaded", False))} for n, p in params.items()}
    (ROOT / f"nvfp4-investigate-{args.tag}-weights.json").write_text(json.dumps(hashes, indent=2))
    selected = {n: p for n, p in params.items() if getattr(p, "_vllm_is_uva_offloaded", False)
                and n.endswith(("w13_weight", "w2_weight"))}
    assert bool(selected) == (args.arm == "patched")
    originals = {n: p.data for n, p in selected.items()}
    resident = {n: p.detach().clone() for n, p in selected.items()}
    torch.cuda.synchronize()
    tokenizer = llm.get_tokenizer()
    if args.stable_alignment:
        from stable_marlin_alignment import install
        install()
    requests = ["Calculate 17 * 23. Give only the number.",
                "用中文用两句话解释为什么天空是蓝色的。",
                "Write a Python function that checks whether a string is a palindrome.",
                "Return a JSON object with keys city and country for Paris."]
    prompts = [tokenizer.apply_chat_template([{"role": "user", "content": t}], tokenize=False,
               add_generation_prompt=True, enable_thinking=False) for t in requests]
    if args.trace:
        trace_first_layer(model, llm, prompts, args.tag)
        return
    captured = []
    compute_logits = model.compute_logits

    def capture(*a, **kw):
        out = compute_logits(*a, **kw)
        captured.append(out.detach().cpu())
        return out

    model.compute_logits = capture
    references = {}
    rows = []
    for residency in ["uva", "uva-repeat", "resident", "resident-repeat", "uva-restored"]:
        use_cpu = residency.startswith("uva")
        for n, p in selected.items():
            p.data = originals[n] if use_cpu else resident[n]
            p._vllm_is_uva_offloaded = use_cpu
        torch.cuda.synchronize()
        for batch in [1, 4]:
            for repeat in range(2):
                captured.clear()
                start = time.perf_counter()
                outputs = llm.generate(prompts[:batch], SamplingParams(temperature=0, max_tokens=1,
                                       ignore_eos=True, logprobs=20), use_tqdm=False)
                assert len(captured) == 1, [list(x.shape) for x in captured]
                logits = captured[0]
                assert torch.isfinite(logits).all()
                key = str(batch)
                row = {"residency": residency, "batch": batch, "repeat": repeat,
                       "seconds": time.perf_counter() - start, "logits_shape": list(logits.shape),
                       "logits_sha256": fingerprint(logits), "tokens": [o.outputs[0].token_ids for o in outputs]}
                if key not in references:
                    references[key] = logits.clone()
                    torch.save(logits, ROOT / f"nvfp4-investigate-{args.tag}-b{batch}-reference.pt")
                row["vs_first_same_batch"] = delta(logits, references[key])
                rows.append(row)
                print("SAME_PROCESS", json.dumps(row), flush=True)
    model.compute_logits = compute_logits
    generations = []
    if args.stable_alignment:
        for residency in ["uva", "resident", "uva-restored"]:
            for n, p in selected.items():
                p.data = resident[n] if residency == "resident" else originals[n]
                p._vllm_is_uva_offloaded = residency != "resident"
            out = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=96, logprobs=5), use_tqdm=False)
            generations.append({"residency": residency, "tokens": [o.outputs[0].token_ids for o in out],
                "text": [o.outputs[0].text for o in out],
                "logprobs": [[{str(k): v.logprob for k, v in s.items()} for s in o.outputs[0].logprobs] for o in out]})
    report = {"arm": args.arm, "tag": args.tag, "weight_hashes": hashes,
              "full_main_python": os.environ.get("PR5090_FULL_MAIN") == "1", "vllm_source": common.vllm.__file__,
              "exact_main_dependencies": os.environ.get("PR5090_EXACT_MAIN") == "1",
              "vllm_version": common.vllm.__version__,
              "torch_version": torch.__version__,
              "flashinfer_version": importlib.metadata.version("flashinfer-python"),
              "offloaded_expert_bytes": sum(p.numel() * p.element_size() for p in selected.values()),
              "requests": requests, "rows": rows,
              "stable_alignment_diagnostic_only": args.stable_alignment, "generations": generations,
              "scope": "One engine; fixed prompts; raw next-token logits; identical weights switched UVA/GPU/UVA"}
    (ROOT / f"nvfp4-investigate-{args.tag}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
