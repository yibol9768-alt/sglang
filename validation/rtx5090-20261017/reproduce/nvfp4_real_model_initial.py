"""Compare the real Qwen3-30B-A3B NVFP4 checkpoint in three engine processes."""
import argparse
import ast
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch
import vllm
import vllm.envs as envs
from vllm.model_executor.model_loader import utils as loader_utils
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_nvfp4 import CompressedTensorsW4A4Nvfp4MoEMethod
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

ROOT = Path(__file__).resolve().parent

def definition(path, name, scope, class_name=None):
    nodes = ast.parse(path.read_text()).body
    if class_name:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name).body
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=["resident", "baseline", "patched"])
    args = ap.parse_args()
    graph = os.environ.get("PR5090_CUDA_GRAPH") == "1"
    tag = args.arm + ("-graph" if graph else "")
    scope = dict(globals())
    CompressedTensorsW4A4Nvfp4MoEMethod.get_post_load_parameter_name = definition(
        ROOT / "sources/nvfp4_method.py", "get_post_load_parameter_name", scope,
        "CompressedTensorsW4A4Nvfp4MoEMethod",
    )
    loader_utils.device_loading_context = definition(
        ROOT / "sources" / f"{'baseline' if args.arm == 'resident' else args.arm}_loader_utils.py",
        "device_loading_context", scope,
    )
    from vllm import LLM, SamplingParams
    start = time.monotonic()
    llm = LLM(
        model=str(ROOT / "models/Qwen3-30B-A3B-NVFP4"), dtype="bfloat16",
        max_model_len=1024, max_num_seqs=4, max_num_batched_tokens=1024,
        enforce_eager=not graph,
        compilation_config={"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4]} if graph else 0,
        gpu_memory_utilization=0.80,
        kv_cache_memory_bytes=512 * 1024**2,
        enable_prefix_caching=False, attention_backend="TRITON_ATTN", moe_backend="marlin",
        cpu_offload_gb=0 if args.arm == "resident" else 1,
        cpu_offload_params={"experts"}, seed=0,
    )
    loaded_s = time.monotonic() - start
    core = llm.llm_engine.engine_core.engine_core
    runner = core.model_executor.driver_worker.worker.model_runner
    model = runner.get_model()
    experts = []
    for name, p in model.named_parameters():
        if name.endswith(("w13_weight", "w2_weight")):
            # Hash each individual layer to bound host staging memory.
            payload = p.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
            experts.append({"name": name, "offloaded": bool(getattr(p, "_vllm_is_uva_offloaded", False)),
                            "bytes": p.numel() * p.element_size(), "sha256": hashlib.sha256(payload).hexdigest()})
            del payload
    offloaded = sum(e["offloaded"] for e in experts)
    assert experts
    assert (offloaded > 0) if args.arm == "patched" else (offloaded == 0)
    requests = [
        "Calculate 17 * 23. Give only the number.",
        "用中文用两句话解释为什么天空是蓝色的。",
        "Write a Python function that checks whether a string is a palindrome.",
        "Return a JSON object with keys city and country for Paris.",
        "Translate into English: 我们正在验证模型推理结果是否一致。",
        "Continue the sequence and explain the rule: 2, 4, 8, 16,",
        "Summarize: A library lets readers borrow books for two weeks. Late returns incur a small fee.",
        "In one sentence, explain the difference between CPU memory and GPU memory.",
    ]
    tokenizer = llm.get_tokenizer()
    prompts = [tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
               add_generation_prompt=True, enable_thinking=False) for text in requests]
    params = SamplingParams(temperature=0, max_tokens=48, logprobs=5)
    outputs = llm.generate(prompts, params, use_tqdm=False)
    assert all(math.isfinite(lp.logprob) for o in outputs for step in o.outputs[0].logprobs for lp in step.values())
    torch.cuda.synchronize()
    durations = []
    for _ in range(3):
        started = time.perf_counter()
        llm.generate(prompts[:4], SamplingParams(temperature=0, max_tokens=24, ignore_eos=True), use_tqdm=False)
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - started)
    report = {
        "arm": args.arm, "model": "RedHatAI/Qwen3-30B-A3B-NVFP4",
        "revision": "a4df9f017835830cbb588cfe72bc4e50294183c0", "vllm": vllm.__version__,
        "mode": "full engine, TP=1, Marlin MoE, fixed batch limit 4", "cuda_graph": graph,
        "load_seconds": loaded_s, "experts": experts,
        "offloaded_expert_bytes": sum(e["bytes"] for e in experts if e["offloaded"]),
        "allocated_bytes": torch.cuda.memory_allocated(), "reserved_bytes": torch.cuda.memory_reserved(),
        "runner_model_memory_usage": getattr(runner, "model_memory_usage", None),
        "max_allocated_bytes": torch.cuda.max_memory_allocated(),
        "repeat_four_prompt_seconds": durations,
        "requests": requests, "text": [o.outputs[0].text for o in outputs],
        "tokens": [o.outputs[0].token_ids for o in outputs],
        "logprobs": [[{str(k): v.logprob for k, v in step.items()} for step in o.outputs[0].logprobs] for o in outputs],
    }
    (ROOT / f"nvfp4-real-{tag}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print("REAL_MODEL_RESULT", json.dumps({k: v for k, v in report.items() if k not in {"experts", "tokens", "logprobs"}}, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
