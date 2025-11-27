#!/bin/bash
# Evaluate all models on OCTA500 6M dataset

echo "========================================="
echo "Evaluating All Models on OCTA500 6M"
echo "========================================="

# Run evaluation
uv run python scripts/evaluate.py \
    --data octa500_6m \
    --output results/octa500_6m \
    --gpu 0

echo ""
echo "✅ Evaluation completed!"
echo "   Results: results/octa500_6m/evaluation_octa500_6m.csv"
