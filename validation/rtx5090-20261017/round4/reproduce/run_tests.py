"""Run the existing complete SM120 file with passive cooling between cases."""
import os
from pathlib import Path
import sys

ROOT = Path(os.environ["PR5090_TASK_ROOT"])
sys.path.insert(0,str(ROOT/'gpu_payload'))
from cool_gpu import wait
import pytest


class Cooling:
    def pytest_runtest_setup(self,item):
        wait()


raise SystemExit(pytest.main(['--noconftest',str(ROOT/'round3/test_sm120_fp8_linear.py'),
    '-q','--tb=short','--junitxml',str(ROOT/'round4/full-fp8-candidate.xml')],plugins=[Cooling()]))
