"""Explore launch schedules using the complete installed PR implementation.

This is a diagnostic harness, not an inference-time override. Quantization,
GEMM reduction K=128 and scaling order stay unchanged. Timings use CUDA
graphs with a 256 MiB cache eviction before each measured call.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(os.environ["PR5090_TASK_ROOT"])
sys.path.insert(0, str(ROOT / "gpu_payload"))
from cool_gpu import wait

import torch
from sglang.kernels.ops.gemm.fp8_kernel import triton_scaled_mm
from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8
from sglang.srt.layers.quantization.fp8_utils import apply_fp8_linear_deterministic


def measure(fn, eviction):
    fn()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        eviction.zero_()
        begin.record()
        output = fn()
        end.record()
    samples = []
    for _ in range(7):
        graph.replay()
        torch.cuda.synchronize()
        samples.append(begin.elapsed_time(end))
    return {"median_ms": statistics.median(samples[2:]), "samples_ms": samples[2:]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("arm")
    args = parser.parse_args()
    torch.manual_seed(291)
    eviction = torch.empty(256 * 1024**2, dtype=torch.uint8, device="cuda")
    a_scale = torch.tensor(.025, device="cuda")
    b_scale = torch.tensor(.02, device="cuda")
    rows = []
    path = ROOT / "round4/schedule-probe.json"
    for n, k in [(6144, 4096), (4096, 4096), (24576, 4096), (4096, 12288)]:
        wait()
        weight = (torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 32).to(torch.float8_e4m3fn).t()
        full_x = torch.randn(256, k, device="cuda", dtype=torch.bfloat16)
        for m in [1, 4, 128, 256]:
            wait()
            x = full_x[-m:].contiguous()
            q, _ = static_quant_fp8(x, a_scale)
            bm = 16 if m <= 16 else 32 if m < 128 else 128
            current = apply_fp8_linear_deterministic(x, weight, b_scale, a_scale)
            quant_time = measure(lambda: static_quant_fp8(x, a_scale), eviction)
            total_time = measure(lambda: apply_fp8_linear_deterministic(x, weight, b_scale, a_scale), eviction)
            for bn, stages in [(64, 2), (32, 2), (64, 3), (128, 3)]:
                wait()
                def gemm(q=q, bm=bm, bn=bn, stages=stages):
                    return triton_scaled_mm(q, weight, a_scale, b_scale, x.dtype,
                        block_size_m=bm, block_size_n=bn, block_size_k=128,
                        use_heuristic=False, num_warps=4, num_stages=stages)
                actual = gemm()
                full_q, _ = static_quant_fp8(full_x, a_scale)
                full = triton_scaled_mm(full_q, weight, a_scale, b_scale, x.dtype,
                    block_size_m=128, block_size_n=bn, block_size_k=128,
                    use_heuristic=False, num_warps=4, num_stages=stages)
                row = {"n": n, "k": k, "m": m, "bm": bm, "bn": bn,
                    "stages": stages, "warps": 4,
                    "equal_to_current_pr": torch.equal(actual, current),
                    "max_abs_vs_current_pr": (actual.float()-current.float()).abs().max().item(),
                    "equal_to_full_batch_suffix": torch.equal(actual, full[-m:]),
                    "quantization": quant_time, "current_total": total_time,
                    "gemm": measure(gemm, eviction)}
                rows.append(row)
                path.write_text(json.dumps({"arm": args.arm, "rows": rows,
                    "method": "normal official-nightly imports; CUDA graph with cache eviction outside timed events; fixed K128 and scaling order"}, indent=2))
                print("PROBE", json.dumps(row), flush=True)
        del weight, full_x, current, actual, full


if __name__ == "__main__":
    main()
