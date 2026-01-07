#!/usr/bin/env python
"""Check multitask model channel configuration."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import autorootcwd
import torch
from src.archs.flow_model import FlowModel

# Load checkpoint
ckpt_path = 'experiments/medsegdiff_flow_multitask/xca/medsegdiff_flow_multitask_xca_20260105_174638/checkpoints/best.ckpt'
print(f'Loading checkpoint: {ckpt_path}')

model = FlowModel.load_from_checkpoint(ckpt_path)
print(f'\n🔍 Model Configuration:')
print(f'  arch_name: {model.hparams.arch_name}')
print(f'  unet type: {type(model.unet).__name__}')

# Check unet architecture
print(f'\n🔍 UNet Architecture:')
print(f'  base_unet.mask_channels: {model.unet.base_unet.mask_channels}')
print(f'  base_unet.input_img_channels: {model.unet.base_unet.input_img_channels}')

# Test forward pass
print('\n🧪 Testing forward pass:')
batch_size = 2
images = torch.randn(batch_size, 1, 64, 64)
t = torch.tensor([0.5, 0.3])

# Try 2-channel input
print('\n1️⃣ Testing 2-channel input (expected for multitask):')
x_2ch = torch.randn(batch_size, 2, 64, 64)
try:
    output = model.unet(x_2ch, t, images)
    print(f'  ✅ Success! Output shape: {output.shape}')
    print(f'  Output channels: {output.shape[1]} (expected: 2 for hard+soft)')
except Exception as e:
    print(f'  ❌ Failed: {e}')

# Try 1-channel input
print('\n2️⃣ Testing 1-channel input (single-task):')
x_1ch = torch.randn(batch_size, 1, 64, 64)
try:
    output = model.unet(x_1ch, t, images)
    print(f'  ✅ Success! Output shape: {output.shape}')
except Exception as e:
    print(f'  ❌ Failed: {str(e)[:150]}')

# Check if training_step would work
print('\n🔍 Simulating training_step:')
print('\n  Current code flow:')
print('    geometry = batch.get("geometry")  # 1-channel SAUNA soft label')
print('    labels = batch.get("label")       # 1-channel hard binary')
print('    noise = torch.randn_like(geometry) # 1-channel noise')
print('    t, xt, ut = flow_matcher.sample(...)')
print('    unet_out = unet(xt, t, images)    # xt is 1-channel')
print('\n  ❌ Problem: Model expects 2-channel input but gets 1-channel!')
print('\n  ✅ Solution: Need to concat [labels, geometry] to make 2-channel target')

print('\n💡 Expected multitask training flow:')
print('    labels = batch["label"]           # (B, 1, H, W) hard binary')
print('    geometry = batch["geometry"]      # (B, 1, H, W) SAUNA soft')
print('    x1 = torch.cat([labels, geometry], dim=1)  # (B, 2, H, W)')
print('    noise = torch.randn_like(x1)      # (B, 2, H, W)')
print('    t, xt, ut = flow_matcher.sample(noise, x1)')
print('    unet_out = unet(xt, t, images)    # xt is (B, 2, H, W)')
print('    # unet_out[:, 0:1] -> hard channel velocity')
print('    # unet_out[:, 1:2] -> soft channel velocity')
