"""Run real tiny Diffusers DC-AE lifecycle regression cases on CPU."""
from pathlib import Path
import importlib.util
import pytest
import sys

ROOT = Path(__file__).resolve().parent
full = sys.argv[2:] == ["all"]
selection = [] if full else ["-k", "TestAutoencoderDCStateLoading"]
if full:
    network_name = "sglang.srt.utils.network"
    network_spec = importlib.util.spec_from_file_location(network_name, ROOT/"sources/network_main.py")
    network_module = importlib.util.module_from_spec(network_spec)
    sys.modules[network_name] = network_module
    network_spec.loader.exec_module(network_module)
    from sglang.multimodal_gen.runtime.loader import component_loaders
    name = "sglang.multimodal_gen.runtime.loader.component_loaders.vae_loader"
    spec = importlib.util.spec_from_file_location(name, ROOT/"sources/vae_loader_main.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    component_loaders.vae_loader = module
fixture_args = [] if full else ["--noconftest"]
raise SystemExit(pytest.main([*fixture_args, str(ROOT/"sources/test_vae_loader.py"),
    *selection, "-q", "--tb=short", "--junitxml",
    str(ROOT/f"autoencoder-tests-{sys.argv[1]}{'-all' if full else ''}.xml")]))
