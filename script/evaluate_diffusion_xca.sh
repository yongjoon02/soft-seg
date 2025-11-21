#!/bin/bash
# Evaluate all diffusion models on XCA dataset

# Models to evaluate
MODELS="segdiff,medsegdiff,berdiff,colddiff,maskdiff,maskdiff_v2"

echo "Evaluating all diffusion models on XCA dataset..."

# Run XCA-specific evaluation script
uv run python script/evaluate_diffusion_xca.py \
    --models "${MODELS}" \
    --output_dir results/xca_diffusion \
    > logs/evaluate_diffusion_xca.log 2>&1 &

echo "Started evaluation (PID: $!)"
echo "Monitor progress with: tail -f logs/evaluate_diffusion_xca.log"
