#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=${PR5090_TASK_ROOT:-/home/liuyibo/codex-workspaces/pr5090}
export PR5090_TASK_ROOT="$TASK_ROOT"
RUN_TAG=$1
MODEL_ARM=$2
MODEL_SCRIPT=$3
[[ "$RUN_TAG" =~ ^[A-Za-z0-9_-]+$ ]]
if [ "${PR5090_MEMORY_SCOPED:-0}" != 1 ]; then
    exec systemd-run --user --quiet --wait --pipe --collect \
        --unit="codex-pr5090-r3-$RUN_TAG-$(date +%s)-$$" \
        -p MemoryMax=8G -p MemorySwapMax=256M -p RuntimeMaxSec=3000 \
        --setenv=PR5090_MEMORY_SCOPED=1 \
        --setenv="PR5090_TASK_ROOT=$TASK_ROOT" \
        --setenv="PR5090_CUDA_GRAPH=${PR5090_CUDA_GRAPH:-0}" \
        bash "$TASK_ROOT/round3/run_sglang_full.sh" "$@"
fi
exec 9>"$TASK_ROOT/gpu-validation.lock"
flock 9
export PATH="$TASK_ROOT/envs/sglang-main-20261021/bin:$PATH"
export CUDA_HOME="$TASK_ROOT/round3/cuda130/nvidia/cu13"
export CPATH="$TASK_ROOT/tools/python-headers/extracted/usr/include:$TASK_ROOT/tools/python-headers/extracted/usr/include/python3.12"
export LIBRARY_PATH="$TASK_ROOT/tools/cuda-link-libs"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:/usr/lib/wsl/lib"
export MAX_JOBS=1
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
unset PYTHONPATH
cd "$TASK_ROOT/gpu_payload"
python cool_gpu.py
python "$TASK_ROOT/round3/prepare_sglang_full_arm.py" "$MODEL_ARM"
exec python gpu_guard.py --log "$TASK_ROOT/round3/$RUN_TAG-thermal.json" --timeout 2400 \
    -- python "$MODEL_SCRIPT" "$MODEL_ARM" "${@:4}"
