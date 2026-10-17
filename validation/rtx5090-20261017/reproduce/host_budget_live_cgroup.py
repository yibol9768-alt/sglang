"""Use delegated Linux cgroups and real anonymous memory for the ancestor case."""
import importlib.util
import json
import mmap
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
GIB = 1024**3

def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "sources" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def write(path, value):
    path.write_text(str(value))

def main():
    baseline = load("host_budget_baseline_live", "sglang_host_memory_budget_baseline.py")
    patched = load("host_budget_patched_live", "sglang_host_memory_budget.py")
    service = Path("/sys/fs/cgroup") / Path("/proc/self/cgroup").read_text().strip().split(":")[-1].lstrip("/")
    control = service / "controller"
    control.mkdir()
    write(control / "cgroup.procs", os.getpid())
    write(service / "cgroup.subtree_control", "+memory")
    parent = service / "nested-parent"
    parent.mkdir()
    write(parent / "memory.max", 2457 * 1024**2)
    write(parent / "cgroup.subtree_control", "+memory")
    leaf, sibling = parent / "inference", parent / "sibling"
    leaf.mkdir(); sibling.mkdir()
    write(leaf / "memory.max", 2252 * 1024**2)
    write(sibling / "memory.max", 640 * 1024**2)
    child = subprocess.Popen([sys.executable, __file__, "--sibling", str(sibling)], stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "ready"
        write(leaf / "cgroup.procs", os.getpid())
        base_cap, base_usage = baseline.cgroup_memory_limit_bytes()
        new_cap, new_usage = patched.cgroup_memory_limit_bytes()
        request = 128 * 1024**2
        before = baseline.HostPinBudget().request(component_name="live-cgroup-128MiB", weight_bytes=request)
        after = patched.HostPinBudget().request(component_name="live-cgroup-128MiB", weight_bytes=request)
        report = {
            "level": "live delegated cgroup v2; real 512 MiB sibling anonymous memory; no cgroup-file mocks",
            "parent": {"limit": int((parent/"memory.max").read_text()), "usage": int((parent/"memory.current").read_text())},
            "leaf": {"limit": int((leaf/"memory.max").read_text()), "usage": int((leaf/"memory.current").read_text())},
            "baseline_selected": [base_cap, base_usage], "patched_selected": [new_cap, new_usage],
            "baseline_headroom": base_cap-base_usage, "patched_headroom": new_cap-new_usage,
            "baseline_128MiB_pin_allowed": before, "patched_128MiB_pin_allowed": after,
            "memory_events": (parent/"memory.events").read_text(),
        }
        print(json.dumps(report, indent=2), flush=True)
        assert base_cap == 2252 * 1024**2
        assert new_cap == 2457 * 1024**2
        assert before and not after
        assert base_cap-base_usage > new_cap-new_usage + 250*1024**2
        (ROOT/"host-budget-live-cgroup.json").write_text(json.dumps(report, indent=2))
    finally:
        write(control / "cgroup.procs", os.getpid())
        child.terminate()
        child.wait(timeout=10)
        child.stdout.close()
        leaf.rmdir(); sibling.rmdir(); parent.rmdir()

if __name__ == "__main__":
    if sys.argv[1:2] == ["--sibling"]:
        write(Path(sys.argv[2]) / "cgroup.procs", os.getpid())
        allocation = mmap.mmap(-1, 512 * 1024**2, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        for offset in range(0, len(allocation), 4096):
            allocation[offset] = 1
        print("ready", flush=True)
        signal.pause()
    else:
        main()
