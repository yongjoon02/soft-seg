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

# All models to evaluate
MODELS="csnet,dscnet,medsegdiff,berdiff"

echo ""
echo ">>> Evaluating models: $MODELS"
echo ""

uv run python scripts/evaluate.py \
    --data "$DATASET" \
    --models "$MODELS" \
    --output "results/${DATASET}" \
    --gpu "$GPU" \
    --save-predictions \
    2>&1 | tee "logs/eval_${DATASET}.log"


echo ""
echo "======================================"
echo "All XCA evaluations completed!"
echo "Results saved to: results/${DATASET}/"
echo "======================================"
