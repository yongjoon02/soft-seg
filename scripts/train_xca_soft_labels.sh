#!/bin/bash
# XCA Soft Label Training Script
# Train CSNet with 3 different soft label types: SAUNA, Gaussian, Smoothing

set -e  # Exit on error

cd /home/yongjun/soft-seg
source .venv/bin/activate

echo "============================================================"
echo "XCA Soft Label Training"
echo "============================================================"
echo ""

# GPU assignment (modify as needed)
GPU_SAUNA=1
GPU_GAUSSIAN=3
GPU_SMOOTH=4

# Train SAUNA
echo "[1/3] Training with SAUNA soft labels on GPU $GPU_SAUNA..."
CUDA_VISIBLE_DEVICES=$GPU_SAUNA python scripts/train.py \
    --config configs/supervised/xca/csnet_label_sauna.yaml &
PID_SAUNA=$!

# Train Gaussian
echo "[2/3] Training with Gaussian soft labels on GPU $GPU_GAUSSIAN..."
CUDA_VISIBLE_DEVICES=$GPU_GAUSSIAN python scripts/train.py \
    --config configs/supervised/xca/csnet_label_gaussian.yaml &
PID_GAUSSIAN=$!

# Train Smoothing
echo "[3/3] Training with Label Smoothing soft labels on GPU $GPU_SMOOTH..."
CUDA_VISIBLE_DEVICES=$GPU_SMOOTH python scripts/train.py \
    --config configs/supervised/xca/csnet_label_smooth.yaml &
PID_SMOOTH=$!

echo ""
echo "All training jobs started in background:"
echo "  SAUNA:    PID=$PID_SAUNA (GPU $GPU_SAUNA)"
echo "  Gaussian: PID=$PID_GAUSSIAN (GPU $GPU_GAUSSIAN)"
echo "  Smooth:   PID=$PID_SMOOTH (GPU $GPU_SMOOTH)"
echo ""
echo "Waiting for all jobs to complete..."

# Wait for all jobs
wait $PID_SAUNA
echo "✅ SAUNA training completed"

wait $PID_GAUSSIAN
echo "✅ Gaussian training completed"

wait $PID_SMOOTH
echo "✅ Smoothing training completed"

echo ""
echo "============================================================"
echo "✅ All soft label training completed!"
echo "============================================================"
