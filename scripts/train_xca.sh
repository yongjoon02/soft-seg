#!/bin/bash

# XCA Dataset Training Script
# 4개 모델(CSNet, DSCNet, MedSegDiff, BerDiff)을 순차적으로 학습

set -e

DATASET="xca"
GPU=0

echo "======================================"
echo "XCA Dataset Training"
echo "======================================"
echo "Dataset: $DATASET"
echo "GPU: $GPU"
echo "======================================"

# Supervised Models
SUPERVISED_MODELS=("csnet" "dscnet")
for MODEL in "${SUPERVISED_MODELS[@]}"; do
    echo ""
    echo ">>> Training $MODEL on $DATASET (GPU $GPU)"
    CONFIG="configs/supervised/${DATASET}/${MODEL}.yaml"
    
    if [ ! -f "$CONFIG" ]; then
        echo "Error: Config file not found: $CONFIG"
        continue
    fi
    
    uv run python scripts/train.py \
        --config "$CONFIG" \
        --trainer.devices "[$GPU]" \
        2>&1 | tee "logs/train_${DATASET}_${MODEL}.log"
    
    echo "<<< Completed $MODEL"
done

# Diffusion Models
DIFFUSION_MODELS=("medsegdiff" "berdiff")
for MODEL in "${DIFFUSION_MODELS[@]}"; do
    echo ""
    echo ">>> Training $MODEL on $DATASET (GPU $GPU)"
    CONFIG="configs/diffusion/${DATASET}/${MODEL}.yaml"
    
    if [ ! -f "$CONFIG" ]; then
        echo "Error: Config file not found: $CONFIG"
        continue
    fi
    
    uv run python scripts/train.py \
        --config "$CONFIG" \
        --trainer.devices "[$GPU]" \
        2>&1 | tee "logs/train_${DATASET}_${MODEL}.log"
    
    echo "<<< Completed $MODEL"
done

echo ""
echo "======================================"
echo "All XCA training completed!"
echo "======================================"
