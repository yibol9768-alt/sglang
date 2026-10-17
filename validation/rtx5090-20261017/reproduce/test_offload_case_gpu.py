"""Run the proposed regression body with real vLLM CUDA/UVA operations.

Loads only the changed definitions from the main checkout onto the installed
wheel; this is a focused integration probe, not the entire repository suite.
"""
import ast
from contextlib import contextmanager
import json
from pathlib import Path

import pytest
import torch
from torch import nn
import vllm.envs as envs
from vllm.model_executor.offloader import UVAOffloader
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_nvfp4 import CompressedTensorsW4A4Nvfp4MoEMethod
from vllm.model_executor.utils import replace_parameter
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

ROOT = Path(__file__).resolve().parent

def definitions(path, names, scope, class_name=None):
    tree = ast.parse(path.read_text())
    nodes = tree.body
    if class_name:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name).body
    selected = [n for n in nodes if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(selected) == len(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), scope)

def main():
    scope = dict(globals())
    definitions(ROOT/"sources/nvfp4_method.py", {"get_post_load_parameter_name"}, scope, "CompressedTensorsW4A4Nvfp4MoEMethod")
    CompressedTensorsW4A4Nvfp4MoEMethod.get_post_load_parameter_name = scope["get_post_load_parameter_name"]
    definitions(ROOT/"sources/test_cpu_offload.py", {"_is_offloaded", "test_nvfp4_expert_rename_preserves_cpu_offload"}, scope)
    results = []
    for arm in ["baseline", "patched"]:
        definitions(ROOT/f"sources/{arm}_loader_utils.py", {"device_loading_context"}, scope)
        for disable_uva in [False, True]:
            try:
                with pytest.MonkeyPatch.context() as monkeypatch:
                    scope["test_nvfp4_expert_rename_preserves_cpu_offload"](monkeypatch, disable_uva)
                passed = True
            except AssertionError as error:
                if arm == "patched":
                    raise
                passed = False
            results.append({"arm": arm, "uva_enabled": not disable_uva, "passed": passed})
    assert [r["passed"] for r in results] == [False, False, True, True]
    print(json.dumps(results, indent=2), flush=True)
    (ROOT/"offload-regression-gpu-results.json").write_text(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
