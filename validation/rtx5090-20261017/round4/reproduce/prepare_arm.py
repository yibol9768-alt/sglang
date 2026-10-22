"""Install complete source files into this task's isolated nightly runtime."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sysconfig

root = Path(os.environ["PR5090_TASK_ROOT"])
parser = argparse.ArgumentParser()
parser.add_argument("arm", choices=["upstream", "current", "candidate"])
args = parser.parse_args()
assert importlib.metadata.version("sglang").startswith("0.5.23.dev20261020")
package = Path(sysconfig.get_paths()["purelib"]) / "sglang"
files = ["srt/layers/quantization/fp8_utils.py", "srt/layers/quantization/modelopt_quant.py", "kernels/ops/gemm/fp8_kernel.py"]
hashes = {}
for relative in files:
    current = root / "round3/nightly-sources/patched" / relative
    target = package / relative
    upstream = root / "round3/nightly-sources/baseline" / relative
    source = {"upstream": upstream, "current": current,
              "candidate": root / "round4/candidate" / relative}[args.arm]
    allowed = {hashlib.sha256(current.read_bytes()).hexdigest()}
    allowed.add(hashlib.sha256(upstream.read_bytes()).hexdigest())
    candidate = root / "round4/candidate" / relative
    if candidate.exists():
        allowed.add(hashlib.sha256(candidate.read_bytes()).hexdigest())
    assert hashlib.sha256(target.read_bytes()).hexdigest() in allowed, f"Unexpected source changed: {target}"
    if target.stat().st_nlink > 1:
        target.unlink()
    shutil.copy2(source, target)
    hashes[relative] = hashlib.sha256(target.read_bytes()).hexdigest()
report = {"arm": args.arm, "files": hashes, "sglang": importlib.metadata.version("sglang")}
(root / f"round4/source-arm-{args.arm}.json").write_text(json.dumps(report, indent=2))
print("SOURCE_ARM", json.dumps(report), flush=True)
