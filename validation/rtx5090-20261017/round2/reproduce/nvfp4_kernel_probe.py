"""Replay real expert inputs inside the engine's forward context."""
import argparse
import json
from pathlib import Path

import torch
import nvfp4_real_model as common
from nvfp4_investigate import delta

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=["baseline", "patched"])
    args = ap.parse_args()
    scope = vars(common).copy()
    cls = common.CompressedTensorsW4A4Nvfp4MoEMethod
    cls.get_post_load_parameter_name = common.definition(ROOT / "sources/nvfp4_method.py",
        "get_post_load_parameter_name", scope, "CompressedTensorsW4A4Nvfp4MoEMethod")
    common.loader_utils.device_loading_context = common.definition(
        ROOT / "sources" / f"{args.arm}_loader_utils.py", "device_loading_context", scope)
    from vllm import LLM, SamplingParams
    from vllm.model_executor.layers.fused_moe.experts import marlin_moe as marlin
    llm = LLM(model=str(ROOT / "models/Qwen3-30B-A3B-NVFP4"), dtype="bfloat16",
              max_model_len=1024, max_num_seqs=4, max_num_batched_tokens=1024,
              enforce_eager=True, compilation_config=0, gpu_memory_utilization=0.80,
              kv_cache_memory_bytes=512 * 1024**2, enable_prefix_caching=False,
              attention_backend="TRITON_ATTN", moe_backend="marlin", cpu_offload_gb=1,
              cpu_offload_params={"experts"}, seed=0)
    model = llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner.get_model()
    layer = model.model.layers[1].mlp.experts.routed_experts
    method = layer.quant_method
    original_apply = method.apply
    align = marlin.moe_align_block_size
    gemm = marlin.ops.moe_wna16_marlin_gemm
    active = []
    calls = []

    def record_align(*a, **kw):
        result = align(*a, **kw)
        active.append({"align": [t.clone() for t in result]})
        return result

    def record_gemm(*a, **kw):
        out = gemm(*a, **kw)
        active[-1].setdefault("gemm", []).append(out.clone())
        return out

    did_probe = False

    def probe(*a, **kw):
        nonlocal did_probe
        if did_probe:
            return original_apply(*a, **kw)
        did_probe = True
        # Store incoming tensors: the layer object remains shared.
        a = tuple(v.clone() if isinstance(v, torch.Tensor) else v for v in a)
        kw = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in kw.items()}
        weights = {n: getattr(layer, n).data for n in ("w13_weight", "w2_weight")}
        flags = {n: bool(getattr(getattr(layer, n), "_vllm_is_uva_offloaded", False)) for n in weights}
        gpu = {n: p.clone() for n, p in weights.items()}
        marlin.moe_align_block_size = record_align
        marlin.ops.moe_wna16_marlin_gemm = record_gemm
        outputs = []
        try:
            for residency in ["original", "resident", "restored"]:
                for n in weights:
                    getattr(layer, n).data = gpu[n] if residency == "resident" else weights[n]
                    getattr(layer, n)._vllm_is_uva_offloaded = False if residency == "resident" else flags[n]
                for repetition in range(4):
                    out = original_apply(*a, **kw).clone()
                    outputs.append(out)
                    calls.append({"residency": residency, "repetition": repetition})
            fixed = active[0]["align"]
            marlin.moe_align_block_size = lambda *a, **kw: tuple(t.clone() for t in fixed)
            # Still collect GEMMs, with a new record for each expert invocation.
            for residency in ["original", "resident", "restored"]:
                for n in weights:
                    getattr(layer, n).data = gpu[n] if residency == "resident" else weights[n]
                    getattr(layer, n)._vllm_is_uva_offloaded = False if residency == "resident" else flags[n]
                for repetition in range(4):
                    active.append({"align": [t.clone() for t in fixed]})
                    out = original_apply(*a, **kw).clone()
                    outputs.append(out)
                    calls.append({"residency": residency, "repetition": repetition, "frozen_alignment": True})
            for i, (call, out, record) in enumerate(zip(calls, outputs, active, strict=True)):
                call["output_vs_first"] = delta(out, outputs[0])
                call["alignment_vs_first"] = [delta(t, r) for t, r in zip(record["align"], active[0]["align"], strict=True)]
                call["gemm_vs_first"] = [delta(t, r) for t, r in zip(record["gemm"], active[0]["gemm"], strict=True)]
                print("KERNEL_PROBE", json.dumps(call), flush=True)
            (ROOT / f"nvfp4-kernel-probe-{args.arm}.json").write_text(json.dumps({"arm": args.arm,
                "layer": "model.layers.1.mlp.experts", "original_uva_flags": flags, "calls": calls}, indent=2))
            return outputs[0]
        finally:
            for n in weights:
                getattr(layer, n).data = weights[n]
                getattr(layer, n)._vllm_is_uva_offloaded = flags[n]
            marlin.moe_align_block_size = align
            marlin.ops.moe_wna16_marlin_gemm = gemm
            method.apply = original_apply

    method.apply = probe
    tokenizer = llm.get_tokenizer()
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": "Calculate 17 * 23. Give only the number."}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    llm.generate([prompt], SamplingParams(temperature=0, max_tokens=1), use_tqdm=False)
    assert did_probe


if __name__ == "__main__":
    main()
