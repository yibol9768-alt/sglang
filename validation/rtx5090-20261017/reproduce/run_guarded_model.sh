#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/path/to/pr5090
cd "$TASK_ROOT/gpu_payload"
if [ "${PR5090_MEMORY_SCOPED:-0}" != 1 ]; then
    exec systemd-run --user --quiet --wait --pipe --collect \
        --unit="codex-pr5090-run-$(date +%s)-$$" \
        -p MemoryMax=8G -p MemorySwapMax=256M -p RuntimeMaxSec=2400 \
        --setenv=PR5090_MEMORY_SCOPED=1 \
        --setenv="PR5090_CUDA_GRAPH=${PR5090_CUDA_GRAPH:-0}" \
        --setenv="PR5090_FP8_TUNING_CONFIRM=${PR5090_FP8_TUNING_CONFIRM:-0}" \
        bash "$0" "$@"
fi
exec 9>"$TASK_ROOT/gpu-validation.lock"
flock 9
RUNTIME_NAME=$1
MODEL_SCRIPT=$2
MODEL_ARM=$3
MODEL_TAG=$MODEL_ARM
if [ "${PR5090_CUDA_GRAPH:-0}" = 1 ]; then
    MODEL_TAG="$MODEL_ARM-graph"
fi
export PATH="$TASK_ROOT/envs/$RUNTIME_NAME/bin:$PATH"
export CUDA_HOME="$TASK_ROOT/envs/vllm/lib/python3.12/site-packages/nvidia/cu13"
export CPATH="$TASK_ROOT/tools/python-headers/extracted/usr/include:$TASK_ROOT/tools/python-headers/extracted/usr/include/python3.12"
export LIBRARY_PATH="$TASK_ROOT/tools/cuda-link-libs"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:/usr/lib/wsl/lib"
export MAX_JOBS=2
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
if [ "$MODEL_SCRIPT" = sana_real_model.py ] || [ "$MODEL_SCRIPT" = live_sana_cgroup.py ]; then
    python "$TASK_ROOT/gpu_payload/prepare_autoencoder_arm.py" patched
    export PR5090_VAE_CHECKPOINT_FIX=1
    python "$TASK_ROOT/gpu_payload/prepare_host_budget_arm.py" "$MODEL_ARM"
elif [ "$MODEL_SCRIPT" = run_vae_loader_pytest.py ]; then
    python "$TASK_ROOT/gpu_payload/prepare_autoencoder_arm.py" "$MODEL_ARM"
elif [ "$RUNTIME_NAME" = sglang ]; then
    python "$TASK_ROOT/gpu_payload/prepare_sglang_arm.py" "$MODEL_ARM"
fi
exec python "$TASK_ROOT/gpu_payload/gpu_guard.py" \
    --log "$TASK_ROOT/logs/${MODEL_SCRIPT%.py}-$MODEL_TAG-thermal.json" --timeout 1800 \
    -- python "$TASK_ROOT/gpu_payload/$MODEL_SCRIPT" "$MODEL_ARM" "${@:4}"
