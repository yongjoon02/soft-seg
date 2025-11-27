#!/bin/bash
# Evaluate all models on ROSSA dataset

echo "========================================="
echo "Evaluating All Models on ROSSA"
echo "========================================="

# Run evaluation
uv run python scripts/evaluate.py \
    --data rossa \
    --output results/rossa \
    --gpu 0

echo ""
echo "✅ Evaluation completed!"
echo "   Results: results/rossa/evaluation_rossa.csv"
