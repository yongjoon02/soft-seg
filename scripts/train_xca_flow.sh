#!/bin/bash
# XCA Flow Model Training Script
# Flow matching: image + noise -> geometry

set -e


# GPU Configuration (default: 0, can override with --gpu N)
GPU=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            GPU="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done
export CUDA_VISIBLE_DEVICES=${GPU}

# Configuration
CONFIG="configs/flow/xca/flow.yaml"

# 로그 디렉토리 및 파일 경로 (logs/로 변경)
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/flow_model_train.log"
mkdir -p "${LOG_DIR}"

echo "======================================"
echo "XCA Flow Model Training"
echo "======================================"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Config: ${CONFIG}"
echo "Log: ${LOG_FILE}"
echo "======================================"

# Check if running in background mode
if [[ "$1" == "--background" || "$1" == "-b" ]]; then
    echo "🚀 Starting Flow model training in background..."
    nohup bash -c "source .venv/bin/activate && CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} uv run python scripts/train.py --config ${CONFIG}" > "${LOG_FILE}" 2>&1 &
    PID=$!
    echo "   PID: ${PID}"
    echo ""
    echo "Monitor with: tail -f ${LOG_FILE}"
    echo "Check status: ps -p ${PID}"
else
    echo "🚀 Starting Flow model training..."
    echo ""
    source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} uv run python scripts/train.py --config "${CONFIG}" | tee "${LOG_FILE}"
fi
