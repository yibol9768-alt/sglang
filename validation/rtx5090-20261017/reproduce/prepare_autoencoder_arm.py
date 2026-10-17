"""Select an AutoencoderDC source version in the private SGLang environment."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys
import sysconfig

ROOT = Path(__file__).resolve().parent
target = Path(sysconfig.get_paths()["purelib"]) / "sglang/multimodal_gen/runtime/models/vaes/autoencoder_dc.py"
backup = ROOT.parent / "tools/sglang-autoencoder-original.py"
if not backup.exists():
    shutil.copy2(target, backup)
    def digest(path):
        return hashlib.sha256(ast.dump(ast.parse(path.read_text()), include_attributes=False).encode()).hexdigest()
    report = {"wheel": digest(backup), "main": digest(ROOT/"sources/autoencoder_dc_baseline.py")}
    report["equal"] = report["wheel"] == report["main"]
    (ROOT/"autoencoder-main-wheel-equivalence.json").write_text(json.dumps(report, indent=2))
    assert report["equal"], report
arm = sys.argv[1]
assert arm in {"baseline", "patched"}
shutil.copy2(ROOT/"sources"/f"autoencoder_dc_{arm}.py", target)
print("AUTOENCODER_SOURCE_ARM", arm, flush=True)
