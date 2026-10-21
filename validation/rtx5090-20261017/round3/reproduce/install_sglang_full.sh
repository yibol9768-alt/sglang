#!/usr/bin/env bash
set -euo pipefail
: "${PR5090_TASK_ROOT:?Set PR5090_TASK_ROOT to your isolated validation directory}"
TASK_ROOT=$PR5090_TASK_ROOT
UV=${PR5090_UV:-uv}
export UV_LINK_MODE=copy
export UV_CONCURRENT_DOWNLOADS=4
export UV_CONCURRENT_BUILDS=1
export MAX_JOBS=1
RUNTIME="$TASK_ROOT/envs/sglang-main-20261021"
mkdir -p "$TASK_ROOT/round3"
"$UV" venv --python 3.12 "$RUNTIME"
"$UV" pip install --python "$RUNTIME/bin/python" --prerelease=allow \
    'https://github.com/sgl-project/whl/releases/download/nightly-2026-10-20-55b2aef7ad/sglang-0.5.23.dev20261020%2Bg55b2aef7ad-cp312-cp312-manylinux_2_39_x86_64.whl#sha256=98f986759bf77b6322f68cddc1eb4c97dec1888e7c26757ecb9e766e73e007e2' \
    pytest parameterized expecttest
# Torch uses CUDA runtime 13.0. Pin an external, matching compiler toolchain.
"$UV" pip install --python "$RUNTIME/bin/python" --target "$TASK_ROOT/round3/cuda130" --no-deps \
    nvidia-cuda-nvcc==13.0.88 nvidia-cuda-crt==13.0.88 nvidia-nvvm==13.0.88 \
    nvidia-cuda-cccl==13.0.85 nvidia-cuda-runtime==13.0.96
"$RUNTIME/bin/python" - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["PR5090_TASK_ROOT"])
src = root / "envs/sglang-main-20261021/lib/python3.12/site-packages/nvidia/cu13"
dst = root / "round3/cuda130/nvidia/cu13"
for folder in ["include", "lib"]:
    for source in (src / folder).rglob("*"):
        target = dst / source.relative_to(src)
        if source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif not target.exists():
            target.symlink_to(source)
PY
