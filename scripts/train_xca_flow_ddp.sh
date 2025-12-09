#!/bin/bash
# XCA Flow Model Training Script with DDP (Multi-GPU)
# Flow matching: image + noise -> geometry

set -e

# GPU Configuration (default: all available, can override with --gpus "0,1,2,3")
GPUS=""
NUM_GPUS=2  # Default number of GPUs

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# Set CUDA_VISIBLE_DEVICES if specific GPUs are requested
if [[ -n "${GPUS}" ]]; then
    export CUDA_VISIBLE_DEVICES=${GPUS}
    # Count number of GPUs from comma-separated list
    NUM_GPUS=$(echo "${GPUS}" | tr ',' '\n' | wc -l)
fi

# Configuration
CONFIG="configs/flow/xca/flow_ddp.yaml"

# 로그 디렉토리 및 파일 경로
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/flow_model_ddp_train.log"
mkdir -p "${LOG_DIR}"

echo "======================================"
echo "XCA Flow Model Training (DDP)"
echo "======================================"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-all available}"
echo "Num GPUs: ${NUM_GPUS}"
echo "Config: ${CONFIG}"
echo "Log: ${LOG_FILE}"
echo "======================================"

# Check if running in background mode
if [[ "$1" == "--background" || "$1" == "-b" ]]; then
    echo "🚀 Starting Flow model DDP training in background..."
    nohup bash -c "source .venv/bin/activate && uv run python scripts/train.py --config ${CONFIG}" > "${LOG_FILE}" 2>&1 &
    PID=$!
    echo "   PID: ${PID}"
    echo ""
    echo "Monitor with: tail -f ${LOG_FILE}"
    echo "Check status: ps -p ${PID}"
else
    echo "🚀 Starting Flow model DDP training..."
    echo ""
    source .venv/bin/activate
    uv run python scripts/train.py --config "${CONFIG}" | tee "${LOG_FILE}"
fi

