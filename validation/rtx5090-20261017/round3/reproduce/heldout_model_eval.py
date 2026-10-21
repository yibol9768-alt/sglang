"""Pinned, disjoint 128-question check with identical extraction and token caps."""
import argparse
from decimal import Decimal, InvalidOperation
import json
import hashlib
import importlib.metadata
import os
from pathlib import Path
import re
import time
from cool_gpu import wait as cool

ROOT = Path(__file__).resolve().parent


def number(text):
    try:
        return Decimal(text.replace(",", "").strip().rstrip("."))
    except InvalidOperation:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=["baseline", "patched"])
    ap.add_argument("--runtime", choices=["vllm", "sglang"], required=True)
    ap.add_argument("--tag")
    ap.add_argument("--disable-uva", action="store_true")
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--limit", type=int, default=128)
    args = ap.parse_args()
    if args.disable_uva:
        os.environ["VLLM_WEIGHT_OFFLOADING_DISABLE_UVA"] = "1"
    data = json.loads((ROOT / "gsm8k-heldout128.json").read_text())
    examples = data["rows"][:args.limit]
    tag = args.tag or f"{args.runtime}-{args.arm}" + ("-non-uva" if args.disable_uva else "")
    if args.runtime == "vllm":
        import nvfp4_real_model as common
        scope = vars(common).copy()
        if os.environ.get("PR5090_FULL_MAIN") != "1":
            common.CompressedTensorsW4A4Nvfp4MoEMethod.get_post_load_parameter_name = common.definition(
                ROOT / "sources/nvfp4_method.py", "get_post_load_parameter_name", scope,
                "CompressedTensorsW4A4Nvfp4MoEMethod")
            common.loader_utils.device_loading_context = common.definition(
                ROOT / "sources" / f"{args.arm}_loader_utils.py", "device_loading_context", scope)
        from vllm import LLM, SamplingParams
        import vllm
        engine = LLM(model=str(ROOT / "models/Qwen3-30B-A3B-NVFP4"), dtype="bfloat16",
            max_model_len=2048, max_num_seqs=4, max_num_batched_tokens=2048,
            enforce_eager=not args.graph,
            compilation_config={"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4]} if args.graph else 0,
            gpu_memory_utilization=.8,
            kv_cache_memory_bytes=1024**3, enable_prefix_caching=False,
            attention_backend="TRITON_ATTN", moe_backend="marlin", cpu_offload_gb=1,
            cpu_offload_params={"experts"}, seed=0,
            **({"linear_backend": "cutlass"} if os.environ.get("PR5090_FULL_MAIN") == "1" else {}))
        tokenizer = engine.get_tokenizer()
        version = vllm.__version__
        model = engine.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner.get_model()
        expert_params = [(n, p) for n, p in model.named_parameters() if n.endswith(("w13_weight", "w2_weight"))]
        offloaded = sum(p.numel() * p.element_size() for n, p in expert_params
            if p.device.type == "cpu" or getattr(p, "_vllm_is_uva_offloaded", False))
        import torch
        loaded_memory = {"allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(), "max_allocated_bytes": torch.cuda.max_memory_allocated()}
        assert (offloaded > 0) == (args.arm == "patched")
        def generate(prompts):
            result = engine.generate(prompts, SamplingParams(temperature=0, max_tokens=512), use_tqdm=False)
            return [(o.outputs[0].text, o.outputs[0].token_ids, o.outputs[0].finish_reason) for o in result]
    else:
        import sglang as sgl
        from transformers import AutoTokenizer
        model_path = str(ROOT / "models/Qwen3-8B-FP8")
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        engine = sgl.Engine(model_path=model_path, dtype="bfloat16", tp_size=1,
            context_length=2048, mem_fraction_static=.65, max_running_requests=4,
            chunked_prefill_size=8192, attention_backend="triton", enable_deterministic_inference=True,
            disable_cuda_graph=True, disable_overlap_schedule=True, random_seed=0, log_level="warning")
        version = sgl.__version__
        offloaded = None
        def generate(prompts):
            result = engine.generate(prompt=prompts, sampling_params={"temperature": 0, "max_new_tokens": 512})
            return [(o["text"], o.get("output_ids"), o["meta_info"].get("finish_reason")) for o in result]
    prompts = [tokenizer.apply_chat_template([{"role": "user", "content":
        "Solve the problem. Keep the reasoning brief. End with 'Answer: <number>'.\n\n" + e["question"]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False) for e in examples]
    assert all(len(tokenizer.encode(p)) + 512 <= 2048 for p in prompts)
    rows = []
    started = time.monotonic()
    report = {"runtime": args.runtime, "version": version, "arm": args.arm, "tag": tag,
        "source": data["source"], "source_sha256": data["whole_file_sha256"], "selection": data["selection"],
        "count": len(examples), "max_output_tokens": 512, "batch_size": 4,
        "disable_uva": args.disable_uva, "offloaded_expert_bytes": offloaded, "rows": rows}
    report["cuda_graph"] = args.graph
    report["model"] = "RedHatAI/Qwen3-30B-A3B-NVFP4" if args.runtime == "vllm" else "nvidia/Qwen3-8B-FP8"
    report["model_revision"] = "a4df9f017835830cbb588cfe72bc4e50294183c0" if args.runtime == "vllm" else "2cebc4c89e25abc17668c81b01dceaf3d8b914d5"
    report["torch_version"] = importlib.metadata.version("torch")
    report["flashinfer_version"] = importlib.metadata.version("flashinfer-python")
    if args.runtime == "vllm":
        report["full_main_python"] = os.environ.get("PR5090_FULL_MAIN") == "1"
        report["exact_main_dependencies"] = os.environ.get("PR5090_EXACT_MAIN") == "1"
        report["vllm_source"] = vllm.__file__
        report["after_load_memory"] = loaded_memory
    elif args.arm == "patched":
        report["helper_source_sha256"] = hashlib.sha256((ROOT / "sources/sglang_fp8_utils.py").read_bytes()).hexdigest()
    try:
        for offset in range(0, len(prompts), 4):
            cool()
            outputs = generate(prompts[offset:offset + 4])
            for index, (example, output) in enumerate(zip(examples[offset:offset + 4], outputs, strict=True), offset):
                text, tokens, finish_reason = output
                matches = re.findall(r"Answer:\s*\*{0,2}\s*\$?([-+]?\d[\d,.]*)", text, flags=re.I)
                predicted = number(matches[-1]) if matches else None
                expected = number(example["answer"].split("####")[-1])
                rows.append({"dataset_index": index + 32, "question": example["question"],
                    "expected": str(expected), "predicted": str(predicted) if predicted is not None else None,
                    "correct": predicted == expected, "text": text, "tokens": tokens, "finish_reason": finish_reason})
            report.update(correct=sum(r["correct"] for r in rows), completed=len(rows),
                missing_final=sum(r["predicted"] is None for r in rows), seconds=time.monotonic() - started)
            (ROOT / f"heldout-eval-{tag}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
            print("HELDOUT_PROGRESS", json.dumps({k: v for k, v in report.items() if k != "rows"}), flush=True)
        if args.benchmark:
            assert args.runtime == "vllm"
            durations = []
            for repeat in range(6):
                cool()
                bench_start = time.perf_counter()
                engine.generate(prompts[:4], SamplingParams(temperature=0, max_tokens=24,
                    ignore_eos=True), use_tqdm=False)
                torch.cuda.synchronize()
                if repeat:
                    durations.append(time.perf_counter() - bench_start)
            report["fixed_work_benchmark"] = {"requests": 4, "forced_output_tokens_each": 24,
                "warmups": 1, "repetitions": 5, "seconds": durations}
            (ROOT / f"heldout-eval-{tag}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        if args.runtime == "sglang":
            engine.shutdown()


if __name__ == "__main__":
    main()
