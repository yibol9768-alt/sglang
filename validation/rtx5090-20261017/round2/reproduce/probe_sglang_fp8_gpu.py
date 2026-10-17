"""Execute SGLang's actual FP8 quantization/GEMM definitions in isolation.

Custom-op registration and runtime platform selection are outside this probe.
No matrix operations, CUDA kernels, or cuBLAS calls are mocked.
"""
import ast
import json
from pathlib import Path
import sys
import types
from typing import Optional, Tuple

import flashinfer
import torch
import triton
import triton.language as tl

ROOT = Path(__file__).resolve().parent

def load_functions(path, names, scope, *, strip_decorators=False):
    tree = ast.parse(path.read_text())
    selected = []
    for name in names:
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
        if strip_decorators:
            node.decorator_list = []
        selected.append(node)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), scope)

def bench(fn, args):
    for _ in range(2):
        fn(*args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(4):
            output = fn(*args)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
    for start, end in zip(starts, ends):
        start.record()
        graph.replay()
        end.record()
    torch.cuda.synchronize()
    return sorted(start.elapsed_time(end) / 4 for start,end in zip(starts,ends))[2]

def main():
    torch.cuda.init()
    torch.backends.cuda.matmul.allow_tf32 = False
    mod = types.ModuleType("sglang_fp8_probe_definitions")
    sys.modules[mod.__name__] = mod
    scope = mod.__dict__
    scope.update({
        "torch":torch, "triton":triton, "tl":tl, "Optional":Optional,
        "Tuple":Tuple, "fp8_dtype":torch.float8_e4m3fn,
        "fp8_min":-448.0,"fp8_max":448.0,
        "fp8_dtype_to_triton":lambda _:tl.float8e4nv,
        "is_arch_support_pdl":lambda:False,
        "_raw_flashinfer_bmm_fp8":flashinfer.bmm_fp8,
    })
    load_functions(ROOT/"sources/sglang_quant_fp8_kernel.py", ["_static_quant_fp8","static_quant_fp8"], scope)
    load_functions(ROOT/"sources/sglang_gemm_fp8_kernel.py", ["is_weak_contiguous","_as_column_scale","scaled_mm_kernel","triton_scaled_mm"], scope)
    load_functions(ROOT/"sources/sglang_fp8_utils.py", ["flashinfer_bmm_fp8"], scope, strip_decorators=True)
    load_functions(ROOT/"sources/sglang_fp8_utils.py", ["apply_fp8_linear_bmm_flashinfer","apply_fp8_linear_deterministic"], scope)
    baseline = scope["apply_fp8_linear_bmm_flashinfer"]
    candidate = scope["apply_fp8_linear_deterministic"]
    quantize = scope["static_quant_fp8"]
    results=[]
    for n,k in [(4096,4096),(16384,5120),(5120,6144),(96,80)]:
        torch.manual_seed(29)
        weight = (torch.randn((n,k),device="cuda",dtype=torch.bfloat16)*32).to(torch.float8_e4m3fn).t()
        a_scale=torch.tensor(0.025,device="cuda")
        b_scale=torch.tensor(0.02,device="cuda")
        for small,large in [(1,64),(8,128),(120,248)]:
            x=torch.randn((large,k),device="cuda",dtype=torch.bfloat16)
            xs=x[-small:].contiguous()
            args=(x,weight,b_scale,a_scale)
            b_full=baseline(*args)
            b_short=baseline(xs,weight,b_scale,a_scale)
            c_full=candidate(*args)
            c_short=candidate(xs,weight,b_scale,a_scale)
            torch.cuda.synchronize()
            q,_=quantize(x,a_scale,repeat_scale=False)
            ref=(q.float()@weight.float())*a_scale*b_scale
            rmse=((c_full.float()-ref).square().mean().sqrt()/ref.square().mean().sqrt()).item()
            row={"n":n,"k":k,"small_m":small,"large_m":large,
                 "baseline_equal":torch.equal(b_full[-small:],b_short),
                 "baseline_max_abs_delta":(b_full[-small:]-b_short).abs().max().item(),
                 "candidate_equal":torch.equal(c_full[-small:],c_short),
                 "candidate_reference_relative_rmse":rmse,
                 "baseline_ms":bench(baseline,args),"candidate_ms":bench(candidate,args)}
            print(json.dumps(row),flush=True)
            assert row["candidate_equal"]
            assert rmse<0.005
            results.append(row)
    output={"level":"Isolated main-source CUDA kernels; PDL disabled consistently in both arms",
            "torch":torch.__version__,"flashinfer":flashinfer.__version__,"gpu":torch.cuda.get_device_name(),"cases":results}
    (ROOT/"sglang-fp8-gpu-evidence.json").write_text(json.dumps(output,indent=2))

if __name__=="__main__":main()
