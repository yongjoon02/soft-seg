#!/bin/bash
# 범용 DDP 학습 스크립트
# 어떤 config 파일이든 DDP로 실행 가능
#
# 사용법:
#   bash scripts/train_ddp.sh --config configs/flow/xca/flow.yaml --gpus "1,2,3,4"
#   bash scripts/train_ddp.sh --config configs/supervised/xca/csnet.yaml --gpus "0,1"
#   bash scripts/train_ddp.sh --config configs/diffusion/octa500_3m/medsegdiff.yaml --gpus "2,3,4,5"

set -e

# 기본값
CONFIG=""
GPUS=""
DEVICES="-1"  # -1 = 모든 visible GPU 사용
STRATEGY="ddp"
PRECISION="16-mixed"
BACKGROUND=false

# 인자 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --config|-c)
            CONFIG="$2"
            shift 2
            ;;
        --gpus|-g)
            GPUS="$2"
            shift 2
            ;;
        --devices|-d)
            DEVICES="$2"
            shift 2
            ;;
        --strategy|-s)
            STRATEGY="$2"
            shift 2
            ;;
        --precision|-p)
            PRECISION="$2"
            shift 2
            ;;
        --background|-b)
            BACKGROUND=true
            shift
            ;;
        --help|-h)
            echo "Usage: bash scripts/train_ddp.sh --config <config.yaml> [options]"
            echo ""
            echo "Options:"
            echo "  --config, -c     Config file path (required)"
            echo "  --gpus, -g       GPU indices (e.g., '0,1,2,3')"
            echo "  --devices, -d    Number of devices or -1 for all (default: -1)"
            echo "  --strategy, -s   DDP strategy (default: ddp)"
            echo "  --precision, -p  Precision (default: 16-mixed)"
            echo "  --background, -b Run in background"
            echo ""
            echo "Examples:"
            echo "  bash scripts/train_ddp.sh -c configs/flow/xca/flow.yaml -g '1,2,3,4'"
            echo "  bash scripts/train_ddp.sh -c configs/supervised/xca/csnet.yaml -g '0,1' -b"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Config 필수 체크
if [[ -z "${CONFIG}" ]]; then
    echo "❌ Error: --config is required"
    echo "Usage: bash scripts/train_ddp.sh --config <config.yaml> --gpus '0,1,2,3'"
    exit 1
fi

# GPU 설정
if [[ -n "${GPUS}" ]]; then
    export CUDA_VISIBLE_DEVICES=${GPUS}
    # GPU 개수 계산
    NUM_GPUS=$(echo "${GPUS}" | tr ',' '\n' | wc -l)
    echo "Using GPUs: ${GPUS} (${NUM_GPUS} devices)"
fi

# 로그 설정
CONFIG_NAME=$(basename "${CONFIG}" .yaml)
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/${CONFIG_NAME}_ddp_train.log"
mkdir -p "${LOG_DIR}"

echo "======================================"
echo "DDP Training"
echo "======================================"
echo "Config: ${CONFIG}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-all available}"
echo "Devices: ${DEVICES}"
echo "Strategy: ${STRATEGY}"
echo "Precision: ${PRECISION}"
echo "Log: ${LOG_FILE}"
echo "======================================"

# DDP 환경변수 설정
export NCCL_P2P_DISABLE=1

# 학습 명령어 (trainer 설정을 환경변수로 override)
# Python에서 config를 로드한 후 이 값들로 override됨
TRAIN_CMD="python scripts/train.py --config ${CONFIG}"

# 실행
if [[ "${BACKGROUND}" == true ]]; then
    echo "🚀 Starting DDP training in background..."
    nohup bash -c "source .venv/bin/activate && \
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
        DDP_DEVICES=${DEVICES} \
        DDP_STRATEGY=${STRATEGY} \
        DDP_PRECISION=${PRECISION} \
        uv run ${TRAIN_CMD}" > "${LOG_FILE}" 2>&1 &
    PID=$!
    echo "   PID: ${PID}"
    echo ""
    echo "Monitor with: tail -f ${LOG_FILE}"
    echo "Check status: ps -p ${PID}"
else
    echo "🚀 Starting DDP training..."
    echo ""
    source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
    DDP_DEVICES=${DEVICES} \
    DDP_STRATEGY=${STRATEGY} \
    DDP_PRECISION=${PRECISION} \
    uv run ${TRAIN_CMD} | tee "${LOG_FILE}"
fi

