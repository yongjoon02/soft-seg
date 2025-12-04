#!/bin/bash
# Train all models on ROSSA dataset with config files

echo "========================================="
echo "Training All Models on ROSSA"
echo "========================================="

# Model list - supervised and diffusion
SUPERVISED_MODELS=("csnet" "dscnet")
DIFFUSION_MODELS=("medsegdiff" "berdiff")
DATASET="rossa"

# GPUs - Use RTX A6000
# Note: CUDA device order differs from nvidia-smi!
# CUDA 2,3,4,5 => nvidia-smi 0,1,2,3 (RTX A6000)
GPUS=(2 3 4 5)

# Train supervised models
idx=0
for MODEL in "${SUPERVISED_MODELS[@]}"; do
    GPU="${GPUS[$idx]}"
    CONFIG="configs/supervised/${DATASET}/${MODEL}.yaml"
    
    echo ""
    echo "🚀 Starting ${MODEL} (supervised) on GPU ${GPU}..."
    
    CUDA_VISIBLE_DEVICES=${GPU} uv run python scripts/train.py \
        --config ${CONFIG} \
        --log-image \
        > logs/train_${DATASET}_${MODEL}.log 2>&1 &
    
    echo "   PID: $!"
    echo "   Config: ${CONFIG}"
    echo "   Log: logs/train_${DATASET}_${MODEL}.log"
    
    idx=$((idx + 1))
    sleep 2
done

# Train diffusion models
for MODEL in "${DIFFUSION_MODELS[@]}"; do
    GPU="${GPUS[$idx]}"
    CONFIG="configs/diffusion/${DATASET}/${MODEL}.yaml"
    
    echo ""
    echo "🚀 Starting ${MODEL} (diffusion) on GPU ${GPU}..."
    
    CUDA_VISIBLE_DEVICES=${GPU} uv run python scripts/train.py \
        --config ${CONFIG} \
        --log-image \
        > logs/train_${DATASET}_${MODEL}.log 2>&1 &
    
    echo "   PID: $!"
    echo "   Config: ${CONFIG}"
    echo "   Log: logs/train_${DATASET}_${MODEL}.log"
    
    idx=$((idx + 1))
    sleep 2
done

echo ""
echo "========================================="
echo "✅ All training processes started!"
echo "========================================="
echo ""
echo "Monitor progress:"
echo "  - TensorBoard: http://localhost:6006"
echo "  - GPU usage: watch -n 1 nvidia-smi"
echo "  - Logs: tail -f logs/train_${DATASET}_*.log"
echo ""
echo "Stop all training:"
echo "  pkill -f train.py"
