"""Collect the actual new SGLang cases against the task-private runtime."""
import argparse
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("arm", choices=["baseline", "patched"])
args = ap.parse_args()
raise SystemExit(pytest.main(["--noconftest", str(ROOT / "sources/test_sm120_fp8_linear.py"),
    "-k", "test_deterministic_modelopt_fp8_preserves_cached_suffix", "-q", "--tb=short",
    "--junitxml", str(ROOT / f"sglang-fp8-native-{args.arm}.xml")]))
