"""Check whether varying only the row tile preserves exact FP8 reductions."""
import json
from pathlib import Path
import sys
import types
from typing import Optional, Tuple
import torch
import triton
import triton.language as tl
from cool_gpu import wait as cool
from probe_sglang_fp8_gpu import load_functions

ROOT = Path(__file__).resolve().parent
mod = types.ModuleType("fp8_row_tile_probe")
sys.modules[mod.__name__] = mod
scope = mod.__dict__
scope.update(torch=torch, triton=triton, tl=tl, Optional=Optional, Tuple=Tuple,
    fp8_dtype=torch.float8_e4m3fn, fp8_min=-448., fp8_max=448.,
    fp8_dtype_to_triton=lambda _: tl.float8e4nv, is_arch_support_pdl=lambda: False)
load_functions(ROOT / "sources/sglang_quant_fp8_kernel.py", ["_static_quant_fp8", "static_quant_fp8"], scope)
load_functions(ROOT / "sources/sglang_gemm_fp8_kernel.py",
    ["is_weak_contiguous", "_as_column_scale", "scaled_mm_kernel", "triton_scaled_mm"], scope)


def time_graph(fn):
    fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(20):
            fn()
    graph.replay()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(5):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / 100


rows = []
torch.manual_seed(734)
for n, k in [(6144, 4096), (4096, 4096), (28672, 4096), (4096, 14336), (96, 80)]:
    weight = (torch.randn((n, k), dtype=torch.bfloat16, device="cuda") * 32).to(torch.float8_e4m3fn).t()
    x = torch.randn((512, k), dtype=torch.bfloat16, device="cuda")
    a_scale, b_scale = torch.tensor(.025, device="cuda"), torch.tensor(.02, device="cuda")
    q, _ = scope["static_quant_fp8"](x, a_scale, repeat_scale=False)
    def gemm(qinput, tile):
        return scope["triton_scaled_mm"](qinput, weight, a_scale, b_scale, torch.bfloat16,
            block_size_m=tile, block_size_n=64, block_size_k=128,
            use_heuristic=False, num_warps=4, num_stages=2)
    reference = gemm(q, 32)
    for m in [1, 4, 8, 64, 248, 512]:
        cool()
        inp = q[-m:].contiguous()
        for tile in [16, 32, 64, 128]:
            out = gemm(inp, tile)
            row = {"n": n, "k": k, "m": m, "row_tile": tile,
                "equal_to_fixed32_full_batch_suffix": torch.equal(out, reference[-m:]),
                "max_abs_vs_fixed32": (out.float() - reference[-m:].float()).abs().max().item(),
                "kernel_ms": time_graph(lambda: gemm(inp, tile))}
            rows.append(row)
            print("M_TILE_PROBE", json.dumps(row), flush=True)
(ROOT / "fp8-row-tile-probe.json").write_text(json.dumps({"scope": "Exploratory; no production change implied", "rows": rows}, indent=2))
