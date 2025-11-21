#!/bin/bash
# Train maskdiff_v2 (with SDF) on XCA dataset

# Base config
CONFIG="configs/xca_diffusion_models.yaml"

# Check if checkpoint exists
CKPT_PATH="lightning_logs/xca/maskdiff_v2/checkpoints/last.ckpt"

if [ -f "$CKPT_PATH" ]; then
    echo "⏭️  Skipping maskdiff_v2 - checkpoint already exists: $CKPT_PATH"
    exit 0
fi

echo "🚀 Starting training for maskdiff_v2 (with SDF) on XCA dataset..."
echo "   - Model: MaskDiff v2 (Mask + Signed Distance Function)"
echo "   - Dataset: XCA (28 train / 7 val / 13 test)"
echo "   - Config: ${CONFIG}"
echo "   - Output: lightning_logs/xca/maskdiff_v2/"
echo ""

# Run training
uv run python script/train_diffusion_models.py fit \
    --config ${CONFIG} \
    --arch_name maskdiff_v2 \
    > logs/train_xca_diffusion_maskdiff_v2.log 2>&1 &

PID=$!
echo "✓ Started maskdiff_v2 (PID: ${PID})"
echo ""
echo "=================================================="
echo "Monitor progress with:"
echo "  tail -f logs/train_xca_diffusion_maskdiff_v2.log"
echo ""
echo "Training details:"
echo "  - 2 channel output: binary mask + SDF"
echo "  - Combined loss: mask + SDF + consistency"
echo "  - Better boundary accuracy expected"
echo "=================================================="
