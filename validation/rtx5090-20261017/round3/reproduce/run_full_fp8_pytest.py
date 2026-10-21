"""Run the complete changed test file against the official-nightly package."""
import argparse
import os
from pathlib import Path

import pytest

from cool_gpu import wait

parser = argparse.ArgumentParser()
parser.add_argument("arm", choices=["baseline", "patched"])
args = parser.parse_args()
root = Path(os.environ["PR5090_TASK_ROOT"]) / "round3"


class Cooling:
    def pytest_runtest_setup(self, item):
        wait()


selection = (
    ["-k", "test_deterministic_modelopt_fp8_preserves_cached_suffix"]
    if args.arm == "baseline"
    else []
)
raise SystemExit(
    pytest.main(
        [
            "--noconftest",
            str(root / "test_sm120_fp8_linear.py"),
            *selection,
            "-q",
            "--tb=short",
            "--junitxml",
            str(root / f"full-fp8-{args.arm}.xml"),
        ],
        plugins=[Cooling()],
    )
)
