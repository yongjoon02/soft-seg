#!/bin/bash
# Test script for soft label generation in diffusion models
# This script demonstrates different soft label methods and compares their behavior

echo "========================================"
echo "Soft Label Generation Test Script"
echo "========================================"
echo ""

# Base config
CONFIG="configs/octa500_3m_diffusion_models.yaml"
MODEL="segdiff"
EPOCHS=1
TRAIN_BATCHES=2
VAL_BATCHES=1

# Test 1: Binary labels (baseline)
echo "Test 1: Binary labels (soft_label_type='none')"
echo "----------------------------------------"
python script/train_diffusion_models.py fit \
    --config $CONFIG \
    --model.arch_name $MODEL \
    --model.soft_label_type none \
    --trainer.max_epochs $EPOCHS \
    --trainer.limit_train_batches $TRAIN_BATCHES \
    --trainer.limit_val_batches $VAL_BATCHES \
    2>&1 | grep -E "train/loss|Epoch"
echo ""

# Test 2: Boundary uncertainty
echo "Test 2: Boundary uncertainty soft labels"
echo "----------------------------------------"
python script/train_diffusion_models.py fit \
    --config $CONFIG \
    --model.arch_name $MODEL \
    --model.soft_label_type boundary \
    --model.soft_label_fg_max 11 \
    --trainer.max_epochs $EPOCHS \
    --trainer.limit_train_batches $TRAIN_BATCHES \
    --trainer.limit_val_batches $VAL_BATCHES \
    2>&1 | grep -E "train/loss|Epoch"
echo ""

# Test 3: Thickness uncertainty
echo "Test 3: Thickness uncertainty soft labels"
echo "----------------------------------------"
python script/train_diffusion_models.py fit \
    --config $CONFIG \
    --model.arch_name $MODEL \
    --model.soft_label_type thickness \
    --model.soft_label_thickness_max 13 \
    --trainer.max_epochs $EPOCHS \
    --trainer.limit_train_batches $TRAIN_BATCHES \
    --trainer.limit_val_batches $VAL_BATCHES \
    2>&1 | grep -E "train/loss|Epoch"
echo ""

# Test 4: SAUNA (combined boundary + thickness)
echo "Test 4: SAUNA soft labels (boundary + thickness)"
echo "----------------------------------------"
python script/train_diffusion_models.py fit \
    --config $CONFIG \
    --model.arch_name $MODEL \
    --model.soft_label_type sauna \
    --model.soft_label_fg_max 11 \
    --model.soft_label_thickness_max 13 \
    --model.soft_label_kernel_ratio 0.1 \
    --trainer.max_epochs $EPOCHS \
    --trainer.limit_train_batches $TRAIN_BATCHES \
    --trainer.limit_val_batches $VAL_BATCHES \
    2>&1 | grep -E "train/loss|Epoch"
echo ""

# Test 5: With caching disabled
echo "Test 5: SAUNA with caching disabled"
echo "----------------------------------------"
python script/train_diffusion_models.py fit \
    --config $CONFIG \
    --model.arch_name $MODEL \
    --model.soft_label_type sauna \
    --model.soft_label_cache false \
    --trainer.max_epochs $EPOCHS \
    --trainer.limit_train_batches $TRAIN_BATCHES \
    --trainer.limit_val_batches $VAL_BATCHES \
    2>&1 | grep -E "train/loss|Epoch"
echo ""

echo "========================================"
echo "All tests completed!"
echo "========================================"
echo ""
echo "Expected behavior:"
echo "- Test 1 (none): Should use binary labels (baseline)"
echo "- Test 2 (boundary): Loss may differ from baseline"
echo "- Test 3 (thickness): Loss may differ from baseline"
echo "- Test 4 (sauna): Loss should be different from binary"
echo "- Test 5 (no cache): Same as Test 4 but regenerates every time"
echo ""
echo "Check that:"
echo "1. All tests run without errors"
echo "2. Losses are in reasonable range (~0.5-2.0)"
echo "3. Different soft label methods produce different losses"
