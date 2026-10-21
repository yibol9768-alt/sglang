"""Select complete PR/base files in the dedicated official-nightly runtime."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sysconfig

root = Path(os.environ["PR5090_TASK_ROOT"]) / "round3"
package = Path(sysconfig.get_paths()["purelib"]) / "sglang"
parser = argparse.ArgumentParser()
parser.add_argument("arm", choices=["baseline", "patched"])
args = parser.parse_args()
files = [
    "srt/layers/quantization/fp8_utils.py",
    "srt/layers/quantization/modelopt_quant.py",
    "kernels/ops/gemm/fp8_kernel.py",
    "multimodal_gen/runtime/managers/memory_managers/host_memory_budget.py",
]
assert importlib.metadata.version("sglang").startswith("0.5.23.dev20261020")
assert importlib.metadata.version("torch").split("+")[0] == "2.14.1"
hashes = {}
for relative in files:
    target = package / relative
    expected = root / "nightly-sources/baseline" / relative
    backup = root / "nightly-pristine" / relative
    if not backup.exists():
        assert target.read_bytes() == expected.read_bytes(), relative
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, backup)
    assert backup.read_bytes() == expected.read_bytes(), relative
    source = root / "nightly-sources" / args.arm / relative
    if target.stat().st_nlink > 1:
        target.unlink()
    shutil.copy2(source, target)
    assert source.read_bytes() == target.read_bytes()
    hashes[relative] = hashlib.sha256(target.read_bytes()).hexdigest()
report = {
    "arm": args.arm,
    "official_nightly_commit": "55b2aef7ad",
    "package": str(package),
    "versions": {
        name: importlib.metadata.version(name)
        for name in ["sglang", "torch", "triton", "flashinfer-python", "sglang-kernel"]
    },
    "files": hashes,
    "method": "Unchanged PR production patch applied to complete nightly files; official dependencies and extensions; no AST injection",
}
(root / f"sglang-full-arm-{args.arm}.json").write_text(json.dumps(report, indent=2))
print("SGLANG_FULL_ARM", json.dumps(report), flush=True)
