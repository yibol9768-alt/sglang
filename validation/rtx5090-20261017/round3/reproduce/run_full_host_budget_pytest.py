"""Exercise the existing host-budget suite through the installed nightly package."""
import argparse
import os
from pathlib import Path

import pytest

parser = argparse.ArgumentParser()
parser.add_argument("arm", choices=["baseline", "patched"])
args = parser.parse_args()
root = Path(os.environ["PR5090_TASK_ROOT"]) / "round3"
raise SystemExit(
    pytest.main(
        [
            "--noconftest",
            str(root / "test_host_memory_budget.py"),
            "-q",
            "--tb=short",
            "--junitxml",
            str(root / f"full-host-budget-{args.arm}.xml"),
        ]
    )
)
