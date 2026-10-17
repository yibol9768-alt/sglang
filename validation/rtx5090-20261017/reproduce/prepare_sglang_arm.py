"""Install only candidate definitions into the task-private SGLang environment."""
import argparse
import ast
from pathlib import Path
import shutil
import sysconfig

ROOT = Path(__file__).resolve().parent
PACKAGE = Path(sysconfig.get_paths()["purelib"]) / "sglang"
BACKUP = ROOT.parent / "tools/sglang-wheel-baseline"
FILES = ["srt/layers/quantization/fp8_utils.py", "srt/layers/quantization/modelopt_quant.py", "kernels/ops/gemm/fp8_kernel.py"]

def node_text(path, name, cls=None):
    source = path.read_text()
    nodes = ast.parse(source).body
    if cls:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == cls).body
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    return "\n".join(source.splitlines()[node.lineno-1:node.end_lineno]) + "\n"

ap = argparse.ArgumentParser()
ap.add_argument("arm", choices=["baseline", "patched"])
args = ap.parse_args()
for relative in FILES:
    source, backup = PACKAGE / relative, BACKUP / relative
    backup.parent.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        shutil.copy2(source, backup)
    shutil.copy2(backup, source)
if args.arm == "patched":
    path = PACKAGE / FILES[0]
    source = path.read_text()
    helper = node_text(ROOT / "sources/sglang_fp8_utils.py", "apply_fp8_linear_deterministic")
    source += "\n\n" + helper
    path.write_text(source)
    path = PACKAGE / FILES[1]
    source = path.read_text().replace("    apply_fp8_linear_bmm_flashinfer,", "    apply_fp8_linear_bmm_flashinfer,\n    apply_fp8_linear_deterministic,")
    source = source.replace("from sglang.srt.runtime_context import get_platform", "from sglang.srt.runtime_context import get_exec, get_platform")
    old = node_text(path, "apply", "ModelOptFp8LinearMethod")
    new = node_text(ROOT / "sources/sglang_modelopt_quant.py", "apply", "ModelOptFp8LinearMethod")
    assert source.count(old) == 1
    source = source.replace(old, new)
    path.write_text(source)
    path = PACKAGE / FILES[2]
    source = path.read_text()
    tree = ast.parse(source)
    original = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "scaled_mm_kernel")
    old = "\n".join(source.splitlines()[original.lineno-1:original.end_lineno]) + "\n"
    new = node_text(ROOT / "sources/sglang_gemm_fp8_kernel.py", "scaled_mm_kernel")
    assert source.count(old) == 1
    path.write_text(source.replace(old, new))
print("SGLang private environment arm:", args.arm, flush=True)
