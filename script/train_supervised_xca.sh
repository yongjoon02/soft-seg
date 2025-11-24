#!/bin/bash
# Train supervised models on XCA dataset

# Base config
CONFIG="configs/xca_supervised_models.yaml"

# Models to train with GPU assignment
MODELS=("csnet" "dscnet")
GPUS=(0 1 2 3 4 5 6)  # Assign each model to different GPU

# Run each model in parallel on different GPUs
for i in "${!MODELS[@]}"; do
    model="${MODELS[$i]}"
    gpu="${GPUS[$i]}"
    
    # Check if checkpoint exists
    CKPT_PATH="lightning_logs/xca/${model}/checkpoints/last.ckpt"
    
    if [ -f "$CKPT_PATH" ]; then
        echo "⏭️  Skipping ${model} - checkpoint already exists: $CKPT_PATH"
        continue
    fi
    
    echo "🚀 Starting training for ${model} on GPU ${gpu}..."
    
    # Run in background with specific GPU
    CUDA_VISIBLE_DEVICES=${gpu} python script/train_supervised_models.py fit \
        --config ${CONFIG} \
        --arch_name ${model} \
        > logs/train_xca_${model}.log 2>&1 &
    
    echo "✓ Started ${model} (PID: $!, GPU: ${gpu})"
    
    # Wait a bit to avoid overwhelming the system
    sleep 5
done

echo ""
echo "=================================================="
echo "All models started. Check logs/ directory for outputs."
echo "To monitor: tail -f logs/train_xca_*.log"
echo "=================================================="
