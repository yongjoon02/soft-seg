#!/bin/bash
# Evaluate all models on OCTA500 3M dataset

echo "========================================="
echo "Evaluating All Models on OCTA500 3M"
echo "========================================="

# Run evaluation
uv run python scripts/evaluate.py \
    --data octa500_3m \
    --output results/octa500_3m \
    --gpu 0

echo ""
echo "✅ Evaluation completed!"
echo "   Results: results/octa500_3m/evaluation_octa500_3m.csv"
