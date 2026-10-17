"""End-to-end engine smoke on a local synthetic quantized MoE checkpoint."""
import argparse
import ast
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import torch
import vllm
import vllm.envs as envs
from vllm.model_executor.model_loader import utils as loader_utils
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_nvfp4 import CompressedTensorsW4A4Nvfp4MoEMethod
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

ROOT=Path(__file__).resolve().parent

def load_definition(path,name,scope,class_name=None):
    tree=ast.parse(path.read_text());nodes=tree.body
    if class_name:nodes=next(n for n in nodes if isinstance(n,ast.ClassDef) and n.name==class_name).body
    node=next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),"exec"),scope)
    return scope[name]

def main():
    ap=argparse.ArgumentParser();ap.add_argument("arm",choices=["resident","baseline","patched"])
    args=ap.parse_args();scope=dict(globals())
    resolver=load_definition(ROOT/"sources/nvfp4_method.py","get_post_load_parameter_name",scope,"CompressedTensorsW4A4Nvfp4MoEMethod")
    CompressedTensorsW4A4Nvfp4MoEMethod.get_post_load_parameter_name=resolver
    loader_utils.device_loading_context=load_definition(ROOT/"sources"/f"{'baseline' if args.arm=='resident' else args.arm}_loader_utils.py","device_loading_context",scope)
    from vllm import LLM,SamplingParams
    llm=LLM(
        model=str(ROOT/"models/tiny-qwen3-moe-nvfp4"),skip_tokenizer_init=True,
        dtype="bfloat16",max_model_len=64,max_num_seqs=2,max_num_batched_tokens=64,
        enforce_eager=True,compilation_config=0,gpu_memory_utilization=0.1,
        enable_prefix_caching=False,attention_backend="TRITON_ATTN",moe_backend="marlin",
        cpu_offload_gb=0 if args.arm=="resident" else 1,
        cpu_offload_params={"experts"},seed=0,
    )
    prompts=[{"prompt_token_ids":[1,10,20,30]},{"prompt_token_ids":[1,45,23,11,9,19]}]
    outputs=llm.generate(prompts,SamplingParams(temperature=0,max_tokens=16,ignore_eos=True,logprobs=5),use_tqdm=False)
    core=llm.llm_engine.engine_core.engine_core
    runner=core.model_executor.driver_worker.worker.model_runner
    model=runner.get_model()
    experts=[]
    for name,p in model.named_parameters():
        if name.endswith(("w13_weight","w2_weight")):
            packed=p.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
            experts.append({"name":name,"offloaded":bool(getattr(p,"_vllm_is_uva_offloaded",False)),"bytes":p.numel()*p.element_size(),"sha256":hashlib.sha256(packed).hexdigest()})
    assert len(experts)==4,experts
    if args.arm=="patched":assert all(p["offloaded"] for p in experts)
    else:assert not any(p["offloaded"] for p in experts)
    report={"arm":args.arm,"vllm":vllm.__version__,"level":"Full vLLM engine; synthetic 2-layer Qwen3 MoE NVFP4; eager mode",
        "experts":experts,"tokens":[o.outputs[0].token_ids for o in outputs],
        "logprobs":[[{str(k):v.logprob for k,v in step.items()} for step in o.outputs[0].logprobs] for o in outputs]}
    (ROOT/f"nvfp4-engine-{args.arm}.json").write_text(json.dumps(report,indent=2))
    print("ENGINE_SMOKE",json.dumps({"arm":args.arm,"tokens":report["tokens"],"offloaded_experts":sum(p["offloaded"] for p in experts)}),flush=True)

if __name__=="__main__":main()
