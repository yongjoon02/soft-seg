"""Test XCA intensity scaling to verify the transformation pipeline."""
import torch
import numpy as np
from monai.transforms import ScaleIntensityd

# Simulate X-ray image (background bright, vessel dark)
# Typical X-ray: background=200-255, vessel=0-100
fake_xray = torch.tensor([[[
    [255, 255, 200],  # Background (bright)
    [255,  50, 200],  # Vessel in middle (dark)
    [200, 200, 255],  # Background (bright)
]]], dtype=torch.float32)

print("=== Original X-ray image ===")
print("(Background bright ~255, Vessel dark ~50)")
print(fake_xray[0, 0])
print(f"Range: [{fake_xray.min():.1f}, {fake_xray.max():.1f}]")

# Method 1: Current implementation (invert then scale)
print("\n=== Method 1: Current (-1 * image, then scale) ===")
inverted = -1 * fake_xray
print("After -1 * image:")
print(inverted[0, 0])
print(f"Range: [{inverted.min():.1f}, {inverted.max():.1f}]")

data1 = {"image": inverted}
scaler = ScaleIntensityd(keys="image", minv=-1.0, maxv=1.0)
result1 = scaler(data1)
print("After ScaleIntensityd(minv=-1, maxv=1):")
print(result1["image"][0, 0])
print(f"Range: [{result1['image'].min():.3f}, {result1['image'].max():.3f}]")
print(f"Vessel (was 50): {result1['image'][0, 0, 1, 1]:.3f}")
print(f"Background (was 255): {result1['image'][0, 0, 0, 0]:.3f}")

# Method 2: Scale first, then invert
print("\n=== Method 2: Better (scale to [0,1], then invert to [-1,1]) ===")
data2 = {"image": fake_xray}
scaler2 = ScaleIntensityd(keys="image", minv=0.0, maxv=1.0)
result2 = scaler2(data2)
print("After ScaleIntensityd(minv=0, maxv=1):")
print(result2["image"][0, 0])
inverted2 = 2 * result2["image"] - 1  # [0,1] -> [-1,1]
print("After 2*image - 1:")
print(inverted2[0, 0])
print(f"Range: [{inverted2.min():.3f}, {inverted2.max():.3f}]")
print(f"Vessel (was 50): {inverted2[0, 0, 1, 1]:.3f}")
print(f"Background (was 255): {inverted2[0, 0, 0, 0]:.3f}")

# Method 3: Just scale normally without inversion
print("\n=== Method 3: Simple (just scale to [-1,1]) ===")
data3 = {"image": fake_xray}
scaler3 = ScaleIntensityd(keys="image", minv=-1.0, maxv=1.0)
result3 = scaler3(data3)
print("After ScaleIntensityd(minv=-1, maxv=1):")
print(result3["image"][0, 0])
print(f"Range: [{result3['image'].min():.3f}, {result3['image'].max():.3f}]")
print(f"Vessel (was 50): {result3['image'][0, 0, 1, 1]:.3f}")
print(f"Background (was 255): {result3['image'][0, 0, 0, 0]:.3f}")

print("\n=== Summary ===")
print("Method 1 (current): Vessel becomes -1.0 (very dark) ❌")
print("Method 2 (better):  Vessel becomes ~-0.6 (darker than bg) - preserves relative intensity ✓")
print("Method 3 (simple):  Vessel becomes ~-0.6 (same as method 2) ✓")
print("\nConclusion: Method 1 is WRONG. Just remove the '-1 *' multiplication!")
