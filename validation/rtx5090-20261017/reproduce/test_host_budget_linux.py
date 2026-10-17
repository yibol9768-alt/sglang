"""Run main's host-memory-budget tests on Linux with real installed dependencies."""
import importlib.util
import argparse
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parent
NAME="sglang.multimodal_gen.runtime.managers.memory_managers.host_memory_budget"

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--baseline",action="store_true")
    args=parser.parse_args()
    path=ROOT/"sources"/("sglang_host_memory_budget_baseline.py" if args.baseline else "sglang_host_memory_budget.py")
    spec=importlib.util.spec_from_file_location(NAME,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[NAME]=module
    spec.loader.exec_module(module)
    import sglang.multimodal_gen.runtime.managers.memory_managers as parent
    parent.host_memory_budget=module
    raise SystemExit(pytest.main(["--noconftest",str(ROOT/"sources/test_host_memory_budget.py"),"-q","--tb=short"]))

if __name__=="__main__":main()
