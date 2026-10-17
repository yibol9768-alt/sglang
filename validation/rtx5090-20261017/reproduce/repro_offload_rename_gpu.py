"""Real CUDA/UVA residency probe; no model and no mocked CUDA operations.

The quantizer's rename/repack lifecycle is reproduced on a uint8 weight.
The original loader context and its candidate patch are executed separately.
This establishes the loader/UVA contract, not complete NVFP4 model correctness.
"""
import ast
from contextlib import contextmanager
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import vllm
import vllm.envs as envs
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
from vllm.model_executor.utils import replace_parameter

ROOT = Path(__file__).resolve().parent

def context_from(path):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "device_loading_context")
    scope = {"torch": torch, "contextmanager": contextmanager, "envs": envs, "is_pin_memory_available": is_pin_memory_available, "get_accelerator_view_from_cpu_tensor": get_accelerator_view_from_cpu_tensor}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope["device_loading_context"]

def run(arm):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated()
    # A 64 MiB weight makes physical CUDA residency visible without a model.
    host = torch.randint(0, 256, (8192, 8192), dtype=torch.uint8).pin_memory()
    expected = host.flip(-1)
    layer = torch.nn.Module()
    layer.quant_method = SimpleNamespace(
        get_post_load_parameter_name=lambda n: {"weight_packed": "weight"}.get(n, n)
    )
    layer.weight_packed = torch.nn.Parameter(
        get_accelerator_view_from_cpu_tensor(host), requires_grad=False
    )
    layer.weight_packed._vllm_is_uva_offloaded = True
    assert layer.weight_packed.device.type == "cuda"
    context = context_from(ROOT / "sources" / f"{arm}_loader_utils.py")
    with context(layer, torch.device("cuda")):
        layer.weight = torch.nn.Parameter(layer.weight_packed.data, requires_grad=False)
        del layer.weight_packed
        replace_parameter(layer, "weight", layer.weight.flip(-1))
    torch.cuda.synchronize()
    retained = torch.cuda.memory_allocated() - before
    torch.testing.assert_close(layer.weight.cpu(), expected, rtol=0, atol=0)
    result = {"arm": arm, "offload_marker": bool(getattr(layer.weight, "_vllm_is_uva_offloaded", False)), "retained_torch_cuda_bytes": retained, "weight_bytes": host.numel(), "values_exact": True}
    del layer, host, expected
    gc.collect()
    torch.cuda.empty_cache()
    return result

def main():
    torch.cuda.init()
    results = [run("baseline"), run("patched")]
    assert not results[0]["offload_marker"]
    assert results[0]["retained_torch_cuda_bytes"] >= results[0]["weight_bytes"]
    assert results[1]["offload_marker"]
    assert results[1]["retained_torch_cuda_bytes"] < results[1]["weight_bytes"] // 2
    output = {"level": "Real CUDA/UVA loader lifecycle, no model", "torch": torch.__version__, "vllm": vllm.__version__, "gpu": torch.cuda.get_device_name(), "results": results}
    print(json.dumps(output, indent=2), flush=True)
    (ROOT / "offload-rename-gpu-evidence.json").write_text(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
