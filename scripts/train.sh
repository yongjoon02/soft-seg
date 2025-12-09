#!/bin/bash
# 범용 학습 스크립트
# Usage:
#   bash scripts/train.sh --config <config.yaml> --gpu <N>
#   bash scripts/train.sh -c configs/supervised/xca/csnet.yaml -g 0
#   bash scripts/train.sh -c configs/flow/xca/flow.yaml --background

set -e

cd /home/yongjun/soft-seg
source .venv/bin/activate

# 기본값
CONFIG=""
GPU=""
BACKGROUND=false

# 인자 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --config|-c)
            CONFIG="$2"
            shift 2
            ;;
        --gpu|-g)
            GPU="$2"
            shift 2
            ;;
        --background|-b)
            BACKGROUND=true
            shift 1
            ;;
        --help|-h)
            echo "Usage: bash scripts/train.sh --config <config.yaml> --gpu <N>"
            echo ""
            echo "Required:"
            echo "  --config, -c     Path to config file"
            echo "  --gpu, -g        GPU index"
            echo ""
            echo "Optional:"
            echo "  --background, -b Run in background"
            echo ""
            echo "Examples:"
            echo "  bash scripts/train.sh -c configs/supervised/xca/csnet.yaml -g 0"
            echo "  bash scripts/train.sh -c configs/flow/xca/flow.yaml -g 1 --background"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 필수 인자 확인
if [[ -z "$CONFIG" ]]; then
    echo "❌ Error: --config is required"
    exit 1
fi

if [[ -z "$GPU" ]]; then
    echo "❌ Error: --gpu is required"
    exit 1
fi

# Config 파일 존재 확인
if [[ ! -f "$CONFIG" ]]; then
    echo "❌ Config file not found: $CONFIG"
    exit 1
fi

# Config 이름에서 로그 파일명 추출
CONFIG_NAME=$(basename "${CONFIG%.*}")
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/${CONFIG_NAME}_train.log"
mkdir -p "${LOG_DIR}"

echo "============================================================"
echo "Training"
echo "============================================================"
echo "Config: ${CONFIG}"
echo "GPU:    ${GPU}"
echo "Log:    ${LOG_FILE}"
echo "============================================================"
echo ""

if ${BACKGROUND}; then
    echo "🚀 Starting training in background..."
    nohup bash -c "CUDA_VISIBLE_DEVICES=${GPU} uv run python scripts/train.py --config ${CONFIG}" > "${LOG_FILE}" 2>&1 &
    PID=$!
    echo "   PID: ${PID}"
    echo ""
    echo "Monitor: tail -f ${LOG_FILE}"
else
    echo "🚀 Starting training..."
    CUDA_VISIBLE_DEVICES=${GPU} uv run python scripts/train.py --config "${CONFIG}" | tee "${LOG_FILE}"
fi

