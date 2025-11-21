#!/bin/bash
# Evaluate all supervised models on XCA dataset

# Models to evaluate
MODELS="cenet,csnet,aacaunet,unet3plus,vesselnet,transunet,dscnet"

echo "Evaluating all supervised models on XCA dataset..."

# Run XCA-specific evaluation script
uv run python script/evaluate_supervised_xca.py \
    --models "${MODELS}" \
    --output_dir results/xca \
    > logs/evaluate_supervised_xca.log 2>&1 &

echo "Started evaluation (PID: $!)"
echo "Monitor progress with: tail -f logs/evaluate_supervised_xca.log"
