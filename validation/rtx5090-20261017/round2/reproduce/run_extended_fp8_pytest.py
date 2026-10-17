"""Run the expanded existing-file FP8 regressions on the private runtime."""
import argparse
from pathlib import Path
import pytest
from cool_gpu import wait as cool

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("arm")
parser.add_argument("--all", action="store_true")
args = parser.parse_args()
selection = [] if args.all else ["-k", "test_deterministic"]


class Cooling:
    def pytest_runtest_setup(self, item):
        cool()


raise SystemExit(pytest.main(["--noconftest", str(ROOT / "sources/test_sm120_fp8_linear.py"),
    *selection, "-q", "--tb=short", "--junitxml", str(ROOT / f"fp8-extended-pytest-{args.arm}{'-all' if args.all else ''}.xml")],
    plugins=[Cooling()]))
