"""Run the real SANA model in a delegated hierarchy with a real sibling."""
import json
import mmap
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent

def write(path, value):
    path.write_text(str(value))

def main(arm):
    relative = Path("/proc/self/cgroup").read_text().strip().split(":")[-1].lstrip("/")
    service = Path("/sys/fs/cgroup") / relative
    assert service.name.startswith("codex-pr5090-sana-") and service.name.endswith(".service"), service
    control = service / "controller"
    control.mkdir()
    for _ in range(5):
        for pid in (service/"cgroup.procs").read_text().split():
            try:
                write(control/"cgroup.procs", pid)
            except ProcessLookupError:
                pass
        try:
            write(service/"cgroup.subtree_control", "+memory")
            break
        except OSError:
            time.sleep(.05)
    else:
        raise RuntimeError("Could not enable the delegated memory controller")
    parent = service / "budget-parent"
    parent.mkdir()
    write(parent/"cgroup.subtree_control", "+memory")
    leaf, sibling = parent/"inference", parent/"sibling"
    leaf.mkdir(); sibling.mkdir()
    write(sibling/"memory.max", 640*1024**2)
    holder = subprocess.Popen([sys.executable, __file__, "--sibling", str(sibling)], stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "ready"
        environment = dict(os.environ, PR5090_NESTED_HOST_BUDGET="1", PR5090_CGROUP_LEAF=str(leaf))
        result = subprocess.run([sys.executable, __file__, "--model", arm], env=environment)
        events = (parent/"memory.events").read_text()
        report = {"arm": arm, "sibling_anonymous_bytes": 512*1024**2, "exit_code": result.returncode,
                  "parent_memory_events": events, "outer_memory_events": (service/"memory.events").read_text(),
                  "parent_limit": (parent/"memory.max").read_text().strip(),
                  "leaf_limit": (leaf/"memory.max").read_text().strip(),
                  "setup": "Real cgroup v2; local file cache reclaimed first, limits calibrated once at first HostPinBudget construction to about 2.25 GiB leaf / 2 GiB parent headroom; real 512 MiB sibling; unchanged budget algorithm"}
        (ROOT/f"sana-cgroup-{arm}.json").write_text(json.dumps(report, indent=2))
        print("REAL_SANA_CGROUP", json.dumps(report), flush=True)
        assert dict(line.split() for line in events.splitlines())["oom_kill"] == "0"
        return result.returncode
    finally:
        holder.terminate()
        holder.wait(timeout=5)
        holder.stdout.close()
        if (parent/"cgroup.kill").exists():
            write(parent/"cgroup.kill", 1)
        for directory in (leaf, sibling, parent):
            try:
                directory.rmdir()
            except OSError:
                pass

if __name__ == "__main__":
    if sys.argv[1:2] == ["--sibling"]:
        write(Path(sys.argv[2])/"cgroup.procs", os.getpid())
        allocation = mmap.mmap(-1, 512*1024**2, flags=mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS)
        for offset in range(0, len(allocation), 4096):
            allocation[offset] = 1
        print("ready", flush=True)
        signal.pause()
    elif sys.argv[1:2] == ["--model"]:
        write(Path(os.environ["PR5090_CGROUP_LEAF"])/"cgroup.procs", os.getpid())
        sys.argv = [str(ROOT/"sana_real_model.py"), sys.argv[2]]
        runpy.run_path(str(ROOT/"sana_real_model.py"), run_name="__main__")
    else:
        raise SystemExit(main(sys.argv[1]))
