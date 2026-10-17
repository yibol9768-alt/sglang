"""Explore fixed GEMM schedules while retaining batch-independent reductions."""
import json
import os
from pathlib import Path
import sys
import types
from typing import Optional, Tuple
import torch
import triton
import triton.language as tl
from probe_sglang_fp8_gpu import load_functions, bench

ROOT = Path(__file__).resolve().parent
mod = types.ModuleType("sglang_fp8_fixed_schedule_probe")
sys.modules[mod.__name__] = mod
scope = mod.__dict__
scope.update({"torch": torch, "triton": triton, "tl": tl, "Optional": Optional, "Tuple": Tuple,
    "fp8_dtype": torch.float8_e4m3fn, "fp8_min": -448.0, "fp8_max": 448.0,
    "fp8_dtype_to_triton": lambda _: tl.float8e4nv, "is_arch_support_pdl": lambda: False})
load_functions(ROOT/"sources/sglang_quant_fp8_kernel.py", ["_static_quant_fp8", "static_quant_fp8"], scope)
load_functions(ROOT/"sources/sglang_gemm_fp8_kernel.py", ["is_weak_contiguous", "_as_column_scale", "scaled_mm_kernel", "triton_scaled_mm"], scope)
rows = []
configs = [(32,64,64), (16,64,128), (16,128,128), (32,64,128)]
confirm = os.environ.get("PR5090_FP8_TUNING_CONFIRM") == "1"
if confirm:
    configs = [(32,64,128), (32,64,64)]
torch.manual_seed(29)
for n,k in [(6144,4096),(4096,4096),(28672,4096),(4096,14336)]:
    weight = (torch.randn((n,k), device="cuda", dtype=torch.bfloat16)*32).to(torch.float8_e4m3fn).t()
    a_scale = torch.tensor(.025, device="cuda")
    b_scale = torch.tensor(.02, device="cuda")
    full_x = torch.randn((248,k), device="cuda", dtype=torch.bfloat16)
    full_q,_ = scope["static_quant_fp8"](full_x, a_scale, repeat_scale=False)
    for config in configs:
        def gemm(q):
            return scope["triton_scaled_mm"](q, weight, a_scale, b_scale, torch.bfloat16,
                block_size_m=config[0], block_size_n=config[1], block_size_k=config[2],
                use_heuristic=False, num_warps=4, num_stages=2)
        full = gemm(full_q)
        ref = (full_q.float()@weight.float()) * a_scale * b_scale
        rmse = ((full.float()-ref).square().mean().sqrt()/ref.square().mean().sqrt()).item()
        assert rmse < .005
        for m in [1,8,64,248]:
            q = full_q[-m:].contiguous()
            result = gemm(q)
            assert torch.equal(result, full[-m:]), (n,k,m,config)
            row = {"n":n, "k":k, "m":m, "tile":config, "relative_rmse":rmse,
                   "kernel_ms":bench(gemm,(q,))}
            rows.append(row)
            print(json.dumps(row), flush=True)
(ROOT/("sglang-fixed-schedule-confirmation.json" if confirm else "sglang-fixed-schedule-tuning.json")).write_text(json.dumps({
    "scope":"Exploratory isolated kernels; quantization time excluded; no end-to-end performance claim", "cases":rows},indent=2))
