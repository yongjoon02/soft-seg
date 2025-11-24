#!/usr/bin/env python3
"""Simple test for core model components without Lightning."""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("Testing Core Model Components")
print("=" * 70)

# Test data (diffusion models need [0, 1] range)
dummy_img = torch.rand(2, 1, 224, 224)  # [0, 1] range for images
dummy_mask = torch.rand(2, 1, 224, 224)  # [0, 1] range for masks

print("\n1. Testing CSNet...")
try:
    from src.archs.components.csnet import CSNet
    csnet = CSNet(in_channels=1, num_classes=2)
    out = csnet(dummy_img)
    params = sum(p.numel() for p in csnet.parameters())
    print(f"   ✓ CSNet works! Output: {out.shape}, Params: {params:,}")
except Exception as e:
    print(f"   ✗ CSNet failed: {e}")
    import traceback
    traceback.print_exc()

print("\n2. Testing DSCNet...")
try:
    from src.archs.components.dscnet import DSCNet
    dscnet = DSCNet(in_channels=1, num_classes=2)
    out = dscnet(dummy_img)
    params = sum(p.numel() for p in dscnet.parameters())
    print(f"   ✓ DSCNet works! Output: {out.shape}, Params: {params:,}")
except Exception as e:
    print(f"   ✗ DSCNet failed: {e}")
    import traceback
    traceback.print_exc()

print("\n3. Testing MedSegDiff...")
try:
    from src.archs.components.gaussian_diffusion import create_medsegdiff
    medsegdiff = create_medsegdiff(image_size=224, dim=32, timesteps=100)
    loss = medsegdiff(dummy_mask, dummy_img)
    params = sum(p.numel() for p in medsegdiff.parameters())
    print(f"   ✓ MedSegDiff works! Loss: {loss.item():.4f}, Params: {params:,}")
except Exception as e:
    print(f"   ✗ MedSegDiff failed: {e}")
    import traceback
    traceback.print_exc()

print("\n4. Testing BerDiff...")
try:
    from src.archs.components.binomial_diffusion import create_berdiff
    berdiff = create_berdiff(image_size=224, dim=32, timesteps=100)
    loss = berdiff(dummy_mask, dummy_img)
    params = sum(p.numel() for p in berdiff.parameters())
    print(f"   ✓ BerDiff works! Loss: {loss.item():.4f}, Params: {params:,}")
except Exception as e:
    print(f"   ✗ BerDiff failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("Testing complete!")
print("=" * 70)
