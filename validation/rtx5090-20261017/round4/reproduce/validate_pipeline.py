"""Qualify the stage-only candidate against the current two-stage PR path."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(os.environ["PR5090_TASK_ROOT"])
sys.path.insert(0, str(ROOT / "gpu_payload"))
from cool_gpu import wait

import torch
from sglang.kernels.ops.gemm.fp8_kernel import triton_scaled_mm
from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8
from sglang.srt.layers.quantization.fp8_utils import apply_fp8_linear_deterministic


def main():
    rows = []
    shapes = [(6144,4096), (4096,4096), (24576,4096), (4096,12288),
              (16384,5120), (5120,6144), (96,80), (97,129)]
    for n,k in shapes:
        for dtype in [torch.bfloat16, torch.float16]:
            for seed in [13,97]:
                wait()
                torch.manual_seed(seed)
                a_scale = torch.tensor(.025 if seed==13 else .003, device="cuda")
                b_scale = torch.tensor(.02 if seed==13 else .01, device="cuda")
                weight = (torch.randn(n,k,device="cuda",dtype=dtype)*32).to(torch.float8_e4m3fn).t()
                full_x = torch.randn(256,k*2,device="cuda",dtype=dtype)[:,::2]
                bias = torch.randn(n,device="cuda",dtype=dtype)
                full = apply_fp8_linear_deterministic(full_x,weight,b_scale,a_scale,bias)
                for m in [1,4,16,17,63,127,128,129,256]:
                    wait()
                    x = full_x[-m:]
                    current_q,_ = static_quant_fp8(x.contiguous(),a_scale)
                    bm = 16 if m<=16 else 32 if m<128 else 128
                    previous = triton_scaled_mm(current_q,weight,a_scale,b_scale,dtype,bias,
                        block_size_m=bm,block_size_n=64,block_size_k=128,
                        use_heuristic=False,num_warps=4,num_stages=2)
                    candidate = apply_fp8_linear_deterministic(x,weight,b_scale,a_scale,bias)
                    equal_previous = torch.equal(previous,candidate)
                    equal_suffix = torch.equal(candidate,full[-m:])
                    row = {"n":n,"k":k,"m":m,"dtype":str(dtype),"seed":seed,
                           "strided_input":True,"bias":True,
                           "equal_to_current_pr":equal_previous,
                           "equal_to_full_batch_suffix":equal_suffix,
                           "max_abs_vs_current":(previous.float()-candidate.float()).abs().max().item()}
                    rows.append(row)
                    assert equal_previous and equal_suffix,row
                print("QUALIFIED", n,k,str(dtype),seed,"cases",len(rows),flush=True)
    (ROOT/'round4/pipeline-equality.json').write_text(json.dumps({"cases":rows,"all_exact":True},indent=2))


if __name__ == '__main__':
    main()
