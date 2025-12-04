#!/bin/bash

# XCA Dataset Training Script
# 4개 모델을 병렬로 학습 (각각 다른 GPU)

DATASET="xca"

echo "======================================"
echo "XCA Dataset Training (Parallel)"
echo "======================================"
echo "GPU Assignment:"
echo "  GPU 0 (Blackwell 98GB): medsegdiff"
echo "  GPU 1 (Blackwell 98GB): berdiff"
echo "  GPU 2 (A6000 49GB): csnet"
echo "  GPU 3 (A6000 49GB): dscnet"
echo "======================================"

# Create logs directory
mkdir -p logs

# Diffusion Models on Blackwell GPUs (0, 1)
echo ""
echo "🚀 Starting medsegdiff (diffusion) on GPU 0 (Blackwell)..."
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train.py \
    --config configs/diffusion/${DATASET}/medsegdiff.yaml \
    --log-image \
    > logs/train_${DATASET}_medsegdiff.log 2>&1 &
echo "   PID: $!"
sleep 2

echo "🚀 Starting berdiff (diffusion) on GPU 1 (Blackwell)..."
CUDA_VISIBLE_DEVICES=1 uv run python scripts/train.py \
    --config configs/diffusion/${DATASET}/berdiff.yaml \
    --log-image \
    > logs/train_${DATASET}_berdiff.log 2>&1 &
echo "   PID: $!"
sleep 2

# Supervised Models on A6000 GPUs (2, 3)
echo ""
echo "🚀 Starting csnet (supervised) on GPU 2 (A6000)..."
CUDA_VISIBLE_DEVICES=2 uv run python scripts/train.py \
    --config configs/supervised/${DATASET}/csnet.yaml \
    --log-image \
    > logs/train_${DATASET}_csnet.log 2>&1 &
echo "   PID: $!"
sleep 2

echo "🚀 Starting dscnet (supervised) on GPU 3 (A6000)..."
CUDA_VISIBLE_DEVICES=3 uv run python scripts/train.py \
    --config configs/supervised/${DATASET}/dscnet.yaml \
    --log-image \
    > logs/train_${DATASET}_dscnet.log 2>&1 &
echo "   PID: $!"

echo ""
echo "======================================"
echo "✅ All training processes started!"
echo "======================================"
echo ""
echo "Monitor progress:"
echo "  - TensorBoard: tensorboard --logdir=lightning_logs --port=6006"
echo "  - GPU usage: watch -n 1 nvidia-smi"
echo "  - Logs: tail -f logs/train_${DATASET}_*.log"
echo ""
echo "Stop all training:"
echo "  pkill -f train.py"
