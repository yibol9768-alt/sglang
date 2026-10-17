"""Real serialized NVFP4 expert loading, Marlin repacking and CUDA forward."""
import ast
from contextlib import contextmanager
import gc
import json
from pathlib import Path

import torch
from torch import nn
import vllm
import vllm.envs as envs
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig, FusedMoEParallelConfig, RoutingMethodType,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_nvfp4 import CompressedTensorsW4A4Nvfp4MoEMethod
from vllm.model_executor.offloader import UVAOffloader
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
from vllm.v1.worker.workspace import init_workspace_manager

ROOT=Path(__file__).resolve().parent

def load_definition(path,name,scope,class_name=None):
    tree=ast.parse(path.read_text())
    nodes=tree.body
    if class_name:
        nodes=next(n for n in nodes if isinstance(n,ast.ClassDef) and n.name==class_name).body
    node=next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),"exec"),scope)
    return scope[name]

def run(arm):
    torch.manual_seed(34)
    cfg=FusedMoEConfig(
        num_experts=8,experts_per_token=2,hidden_dim=1024,intermediate_size=1024,
        num_local_experts=8,num_logical_experts=8,activation=MoEActivation.SILU,
        device="cuda",routing_method=RoutingMethodType.TopK,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        in_dtype=torch.bfloat16,moe_backend="marlin",max_num_tokens=64,
    )
    method=CompressedTensorsW4A4Nvfp4MoEMethod(cfg,use_a16=True)
    layer=nn.Module()
    layer.quant_method=method
    layer._expert_routing_tables=lambda:None
    layer.activation=MoEActivation.SILU
    layer.global_num_experts=8
    layer.hidden_size=1024
    layer.intermediate_size_per_partition=1024
    layer.expert_map=None
    layer.apply_router_weight_on_input=False
    with torch.device("cuda"):
        method.create_weights(layer,8,1024,1024,torch.bfloat16)
    with torch.no_grad():
        for name,p in layer.named_parameters():
            if name.endswith("_packed"):
                p.copy_(torch.randint(0,256,p.shape,dtype=p.dtype,device="cuda"))
            elif "global_scale" in name:
                p.fill_(64.0)
            else:
                p.fill_(1.0)
    if arm!="resident":
        offloader=UVAOffloader(1024**3,{"w13_weight_packed","w2_weight_packed"})
        offloader.wrap_modules(iter([layer]))
        assert all(getattr(getattr(layer,n),"_vllm_is_uva_offloaded",False) for n in ["w13_weight_packed","w2_weight_packed"])
    scope=dict(globals())
    context=load_definition(ROOT/"sources"/f"{'baseline' if arm=='resident' else arm}_loader_utils.py","device_loading_context",scope)
    with context(layer,torch.device("cuda")):
        method.process_weights_after_loading(layer)
    markers={n:bool(getattr(getattr(layer,n),"_vllm_is_uva_offloaded",False)) for n in ["w13_weight","w2_weight"]}
    x=torch.randn((8,1024),dtype=torch.bfloat16,device="cuda")/10
    ids=torch.tensor([[0,1],[2,3],[4,5],[6,7],[0,2],[1,3],[4,6],[5,7]],device="cuda",dtype=torch.int32)
    weights=torch.full((8,2),0.5,device="cuda")
    output=method.apply(layer,x,weights,ids,None,None)
    torch.cuda.synchronize()
    assert isinstance(output,torch.Tensor)
    assert torch.isfinite(output).all()
    result={"arm":arm,"backend":method.nvfp4_backend.value,"offload_markers":markers,"shape":list(output.shape),"finite":True}
    cpu_output=output.cpu()
    del layer,method,output
    gc.collect();torch.cuda.empty_cache()
    return result,cpu_output

def main():
    scope=dict(globals())
    resolver=load_definition(ROOT/"sources/nvfp4_method.py","get_post_load_parameter_name",scope,"CompressedTensorsW4A4Nvfp4MoEMethod")
    CompressedTensorsW4A4Nvfp4MoEMethod.get_post_load_parameter_name=resolver
    torch.cuda.set_device(0)
    init_workspace_manager(torch.device("cuda",0))
    with set_current_vllm_config(VllmConfig()):
        rows=[]; reference=None
        for arm in ["resident","baseline","patched"]:
            result,out=run(arm)
            if reference is None:reference=out
            else:torch.testing.assert_close(out,reference,rtol=0,atol=0)
            result["bitwise_matches_resident"]=torch.equal(out,reference)
            print(json.dumps(result),flush=True)
            rows.append(result)
    assert not any(rows[1]["offload_markers"].values())
    assert all(rows[2]["offload_markers"].values())
    report={"level":"Real NVFP4 method, Marlin repack and CUDA forward; synthetic expert weights; no full model", "vllm":vllm.__version__,"torch":torch.__version__,"results":rows}
    (ROOT/"nvfp4-layer-gpu-evidence.json").write_text(json.dumps(report,indent=2))

if __name__=="__main__":main()
