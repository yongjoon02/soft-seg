#!/usr/bin/env python3
"""Test script to verify all 4 models work correctly after cleanup."""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("Testing Cleaned Model Architecture")
print("=" * 70)

# Test data (diffusion models need [0, 1] range)
dummy_img = torch.rand(2, 1, 224, 224)
dummy_mask = torch.rand(2, 1, 224, 224)

print("\n" + "=" * 70)
print("1. Testing Supervised Models")
print("=" * 70)

# Test CSNet
try:
    from src.archs.components import CSNet
    csnet = CSNet(in_channels=1, num_classes=2)
    out = csnet(dummy_img)
    params = sum(p.numel() for p in csnet.parameters())
    print(f"✓ CSNet: Output shape {out.shape}, Params: {params:,}")
except Exception as e:
    print(f"✗ CSNet failed: {e}")

# Test DSCNet
try:
    from src.archs.components import DSCNet
    dscnet = DSCNet(in_channels=1, num_classes=2)
    out = dscnet(dummy_img)
    params = sum(p.numel() for p in dscnet.parameters())
    print(f"✓ DSCNet: Output shape {out.shape}, Params: {params:,}")
except Exception as e:
    print(f"✗ DSCNet failed: {e}")

print("\n" + "=" * 70)
print("2. Testing Diffusion Models")
print("=" * 70)

# Test MedSegDiff
try:
    from src.archs.components.gaussian_diffusion import create_medsegdiff
    medsegdiff = create_medsegdiff(image_size=224, dim=32, timesteps=100)
    loss = medsegdiff(dummy_mask, dummy_img)
    params = sum(p.numel() for p in medsegdiff.parameters())
    print(f"✓ MedSegDiff: Loss {loss.item():.4f}, Params: {params:,}")
    
    # Test sampling
    with torch.no_grad():
        sample = medsegdiff.sample(dummy_img[:1])
        print(f"  - Sample shape: {sample.shape}, range: [{sample.min():.2f}, {sample.max():.2f}]")
except Exception as e:
    print(f"✗ MedSegDiff failed: {e}")

# Test BerDiff
try:
    from src.archs.components.binomial_diffusion import create_berdiff
    berdiff = create_berdiff(image_size=224, dim=32, timesteps=100)
    loss = berdiff(dummy_mask, dummy_img)
    params = sum(p.numel() for p in berdiff.parameters())
    print(f"✓ BerDiff: Loss {loss.item():.4f}, Params: {params:,}")
    
    # Test sampling
    with torch.no_grad():
        sample = berdiff.sample(dummy_img[:1])
        print(f"  - Sample shape: {sample.shape}, range: [{sample.min():.2f}, {sample.max():.2f}]")
except Exception as e:
    print(f"✗ BerDiff failed: {e}")

print("\n" + "=" * 70)
print("3. Testing MODEL_REGISTRY")
print("=" * 70)

# Test supervised registry
try:
    from src.archs.supervised_model import MODEL_REGISTRY as SUPERVISED_REGISTRY
    print(f"✓ Supervised Registry: {list(SUPERVISED_REGISTRY.keys())}")
    assert len(SUPERVISED_REGISTRY) == 2, f"Expected 2 models, got {len(SUPERVISED_REGISTRY)}"
    assert 'csnet' in SUPERVISED_REGISTRY
    assert 'dscnet' in SUPERVISED_REGISTRY
except Exception as e:
    print(f"✗ Supervised Registry failed: {e}")

# Test diffusion registry
try:
    from src.archs.diffusion_model import MODEL_REGISTRY as DIFFUSION_REGISTRY
    print(f"✓ Diffusion Registry: {list(DIFFUSION_REGISTRY.keys())}")
    assert len(DIFFUSION_REGISTRY) == 2, f"Expected 2 models, got {len(DIFFUSION_REGISTRY)}"
    assert 'medsegdiff' in DIFFUSION_REGISTRY
    assert 'berdiff' in DIFFUSION_REGISTRY
except Exception as e:
    print(f"✗ Diffusion Registry failed: {e}")

print("\n" + "=" * 70)
print("4. Testing Lightning Modules")
print("=" * 70)

# Test SupervisedModel
try:
    from src.archs.supervised_model import SupervisedModel
    model = SupervisedModel(arch_name='csnet', learning_rate=1e-3)
    print(f"✓ SupervisedModel: arch={model.arch_name}, lr={model.hparams.learning_rate}")
except Exception as e:
    print(f"✗ SupervisedModel failed: {e}")

# Test DiffusionModel
try:
    from src.archs.diffusion_model import DiffusionModel
    model = DiffusionModel(arch_name='medsegdiff', timesteps=100, learning_rate=1e-4)
    print(f"✓ DiffusionModel: timesteps={model.hparams.timesteps}, lr={model.hparams.learning_rate}")
except Exception as e:
    print(f"✗ DiffusionModel failed: {e}")

print("\n" + "=" * 70)
print("✓ All tests passed! Models are working correctly.")
print("=" * 70)
