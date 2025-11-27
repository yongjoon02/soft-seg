#!/bin/bash

# XCA Dataset Evaluation Script
# 모든 모델의 체크포인트를 평가

set -e

DATASET="xca"
GPU=0

echo "======================================"
echo "XCA Dataset Evaluation"
echo "======================================"
echo "Dataset: $DATASET"
echo "GPU: $GPU"
echo "======================================"

# Supervised Models
SUPERVISED_MODELS=("csnet" "dscnet")
for MODEL in "${SUPERVISED_MODELS[@]}"; do
    echo ""
    echo ">>> Evaluating $MODEL on $DATASET"
    
    CHECKPOINT="lightning_logs/${DATASET}/${MODEL}/checkpoints/best.ckpt"
    
    if [ ! -f "$CHECKPOINT" ]; then
        echo "Warning: Checkpoint not found: $CHECKPOINT"
        echo "Skipping $MODEL"
        continue
    fi
    
    uv run python scripts/evaluate.py \
        --checkpoint "$CHECKPOINT" \
        --data "$DATASET" \
        --output "results/${DATASET}/${MODEL}" \
        --gpu "$GPU" \
        2>&1 | tee "logs/eval_${DATASET}_${MODEL}.log"
    
    echo "<<< Completed $MODEL evaluation"
done

# Diffusion Models
DIFFUSION_MODELS=("medsegdiff" "berdiff")
for MODEL in "${DIFFUSION_MODELS[@]}"; do
    echo ""
    echo ">>> Evaluating $MODEL on $DATASET"
    
    CHECKPOINT="lightning_logs/${DATASET}/${MODEL}/checkpoints/best.ckpt"
    
    if [ ! -f "$CHECKPOINT" ]; then
        echo "Warning: Checkpoint not found: $CHECKPOINT"
        echo "Skipping $MODEL"
        continue
    fi
    
    uv run python scripts/evaluate.py \
        --checkpoint "$CHECKPOINT" \
        --data "$DATASET" \
        --output "results/${DATASET}/${MODEL}" \
        --gpu "$GPU" \
        2>&1 | tee "logs/eval_${DATASET}_${MODEL}.log"
    
    echo "<<< Completed $MODEL evaluation"
done

echo ""
echo "======================================"
echo "All XCA evaluations completed!"
echo "Results saved to: results/${DATASET}/"
echo "======================================"
