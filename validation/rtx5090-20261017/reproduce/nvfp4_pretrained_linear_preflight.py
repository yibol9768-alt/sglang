"""Compile the real checkpoint's dense NVFP4 path before full model startup."""
import json
import os
from pathlib import Path
import tempfile
import torch
from safetensors import safe_open
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import init_distributed_environment, initialize_model_parallel
from vllm.v1.worker.workspace import init_workspace_manager
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsConfig

ROOT = Path(__file__).resolve().parent
model = ROOT/"models/Qwen3-30B-A3B-NVFP4"
prefix = "model.layers.0.self_attn.q_proj"
quant = CompressedTensorsConfig.from_config(json.loads((model/"config.json").read_text())["quantization_config"])
fd, rendezvous = tempfile.mkstemp()
os.close(fd)
init_distributed_environment(world_size=1, rank=0, distributed_init_method="file://"+rendezvous,
                             local_rank=0, backend="nccl")
with safe_open(model/"model-00001-of-00004.safetensors", framework="pt", device="cpu") as checkpoint:
    n, half_k = checkpoint.get_slice(prefix+".weight_packed").get_shape()
    with set_current_vllm_config(VllmConfig()), torch.device("cuda"), torch.no_grad():
        initialize_model_parallel(1, 1)
        init_workspace_manager(torch.device("cuda", 0))
        layer = ReplicatedLinear(2*half_k, n, bias=False, params_dtype=torch.bfloat16,
                                 quant_config=quant, prefix=prefix)
        loaded = []
        for name, parameter in layer.named_parameters():
            source = checkpoint.get_tensor(prefix+"."+name)
            parameter.copy_(source.reshape(parameter.shape))
            loaded.append(name)
        layer.quant_method.process_weights_after_loading(layer)
        outputs = []
        for m in [1, 8, 64]:
            result = layer(torch.randn(m, 2*half_k, dtype=torch.bfloat16)/10)[0]
            assert torch.isfinite(result).all()
            outputs.append({"m":m, "shape":list(result.shape), "max_abs":result.abs().max().item()})
        torch.cuda.synchronize()
        report = {"scope":"one actual pretrained NVFP4 q_proj; full checkpoint model not loaded",
                  "parameters":loaded,"kernel":type(layer.scheme.kernel).__name__,"outputs":outputs}
        print("PRETRAINED_NVFP4_DENSE",json.dumps(report),flush=True)
        (ROOT/"nvfp4-pretrained-dense-preflight.json").write_text(json.dumps(report,indent=2))
