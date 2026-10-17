"""Run the actual changed pytest file and its normal repository fixtures."""
import argparse
import os
from contextlib import contextmanager
from pathlib import Path
import sys

import pytest
import torch
import vllm.envs as envs
from vllm.model_executor.model_loader import utils as loader_utils
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_nvfp4 import CompressedTensorsW4A4Nvfp4MoEMethod
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
from nvfp4_real_model import definition

ROOT = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("arm", choices=["baseline", "patched"])
args = ap.parse_args()
scope = dict(globals())
if os.environ.get("PR5090_EXACT_MAIN") != "1":
    CompressedTensorsW4A4Nvfp4MoEMethod.get_post_load_parameter_name = definition(
        ROOT / "sources/nvfp4_method.py", "get_post_load_parameter_name", scope,
        "CompressedTensorsW4A4Nvfp4MoEMethod",
    )
    loader_utils.device_loading_context = definition(
        ROOT / "sources" / f"{args.arm}_loader_utils.py", "device_loading_context", scope,
    )
repository = ROOT.parent / "selected-vllm"
sys.path.insert(0, str(repository))
raise SystemExit(pytest.main([str(repository / "tests/basic_correctness/cpu_offload/test_cpu_offload.py"),
    "-k", "test_nvfp4_expert_rename_preserves_cpu_offload", "-q", "--tb=short",
    "--junitxml", str(ROOT / f"native-offload-exact-{args.arm}.xml")]))
