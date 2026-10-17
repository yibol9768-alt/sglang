"""Select the main-source host-budget module inside the private environment."""
import argparse
from pathlib import Path
import sysconfig

ROOT = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("arm", choices=["baseline", "patched"])
args = ap.parse_args()
source = ROOT / "sources" / ("sglang_host_memory_budget_baseline.py" if args.arm == "baseline" else "sglang_host_memory_budget.py")
target = Path(sysconfig.get_paths()["purelib"]) / "sglang/multimodal_gen/runtime/managers/memory_managers/host_memory_budget.py"
source_text = source.read_text()
# Observability only, applied identically in both test arms.
source_text += '''

_codex_original_budget_init = HostPinBudget.__init__
_codex_original_budget_request = HostPinBudget.request
_codex_calibrated = False
def _codex_budget_init(self, available_bytes=None):
    global _codex_calibrated
    if os.environ.get("PR5090_NESTED_HOST_BUDGET") == "1" and not _codex_calibrated:
        from pathlib import Path
        import errno
        import gc
        import ctypes
        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        leaf = Path(os.environ["PR5090_CGROUP_LEAF"])
        parent = leaf.parent
        file_cache = int(dict(line.split() for line in (leaf / "memory.stat").read_text().splitlines()).get("file", 0))
        if file_cache:
            try:
                (leaf / "memory.reclaim").write_text(str(file_cache))
            except OSError as error:
                if error.errno != errno.EAGAIN:
                    raise
        leaf_limit = int((leaf / "memory.current").read_text()) + 2304 * 1024**2
        parent_limit = int((parent / "memory.current").read_text()) + 2048 * 1024**2
        assert parent_limit > leaf_limit
        (parent / "memory.max").write_text(str(parent_limit))
        (leaf / "memory.max").write_text(str(leaf_limit))
        print("REAL_HOST_BUDGET_CGROUP", leaf_limit, parent_limit, flush=True)
        _codex_calibrated = True
    _codex_original_budget_init(self, available_bytes)
    print("REAL_HOST_BUDGET_INIT", self.available_bytes, self.reserve_bytes, flush=True)
def _codex_budget_request(self, *, component_name, weight_bytes):
    allowed = _codex_original_budget_request(self, component_name=component_name, weight_bytes=weight_bytes)
    print("REAL_HOST_BUDGET_REQUEST", component_name, weight_bytes, allowed, self.committed_bytes, flush=True)
    return allowed
HostPinBudget.__init__ = _codex_budget_init
HostPinBudget.request = _codex_budget_request
'''
target.write_text(source_text)
print("Private SGLang diffusion host-budget arm:", args.arm, flush=True)
