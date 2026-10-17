#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=${PR5090_TASK_ROOT:-/path/to/pr5090}
cd "$TASK_ROOT/gpu_payload"
RUNTIME_NAME=$1
RUN_TAG=$2
MODEL_SCRIPT=$3
MODEL_ARM=$4
[[ "$RUN_TAG" =~ ^[A-Za-z0-9_-]+$ ]]
if [ "${PR5090_MEMORY_SCOPED:-0}" != 1 ]; then
    exec systemd-run --user --quiet --wait --pipe --collect \
        --unit="codex-pr5090-r2-$RUN_TAG-$(date +%s)-$$" \
        -p MemoryMax=8G -p MemorySwapMax=256M -p RuntimeMaxSec=2400 \
        --setenv=PR5090_MEMORY_SCOPED=1 \
        --setenv="PR5090_TASK_ROOT=$TASK_ROOT" \
        --setenv="PR5090_CUDA_GRAPH=${PR5090_CUDA_GRAPH:-0}" \
        bash "$0" "$@"
fi
exec 9>"$TASK_ROOT/gpu-validation.lock"
flock 9
export PATH="$TASK_ROOT/envs/$RUNTIME_NAME/bin:$PATH"
export CUDA_HOME="$TASK_ROOT/envs/vllm/lib/python3.12/site-packages/nvidia/cu13"
if [ "$RUNTIME_NAME" = vllm-main ]; then
    export CUDA_HOME="$TASK_ROOT/envs/vllm-main/lib/python3.12/site-packages/nvidia/cu13"
fi
export CPATH="$TASK_ROOT/tools/python-headers/extracted/usr/include:$TASK_ROOT/tools/python-headers/extracted/usr/include/python3.12"
export LIBRARY_PATH="$TASK_ROOT/tools/cuda-link-libs"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:/usr/lib/wsl/lib"
export MAX_JOBS=1
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
PYTHON_BIN="$TASK_ROOT/envs/$RUNTIME_NAME/bin/python"
"$PYTHON_BIN" "$TASK_ROOT/gpu_payload/cool_gpu.py"
if [ "$RUNTIME_NAME" = sglang ]; then
    "$PYTHON_BIN" "$TASK_ROOT/gpu_payload/prepare_sglang_arm.py" "$MODEL_ARM"
elif [ "$RUNTIME_NAME" = vllm-main ]; then
    unset PYTHONPATH
    export VLLM_CACHE_ROOT="$TASK_ROOT/tools/exact-main-runtime-cache"
    export PR5090_FULL_MAIN=1
    export PR5090_EXACT_MAIN=1
    "$PYTHON_BIN" "$TASK_ROOT/gpu_payload/prepare_exact_main_arm.py" "$MODEL_ARM"
fi
exec "$PYTHON_BIN" "$TASK_ROOT/gpu_payload/gpu_guard.py" \
    --log "$TASK_ROOT/logs/round2-$RUN_TAG-thermal.json" --timeout 1800 \
    -- "$PYTHON_BIN" "$TASK_ROOT/gpu_payload/$MODEL_SCRIPT" "$MODEL_ARM" "${@:5}"
