#!/usr/bin/env python
"""Check gradient conflict between flow loss and geometry loss."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.archs.flow_model import FlowModel
from src.data.xca import XCADataset

# Load model
config_path = 'configs/flow/xca/flow_sauna_medsegdiff.yaml'
ckpt_path = 'experiments/medsegdiff_flow/xca/medsegdiff_flow_xca_20260106_143543/checkpoints/best.ckpt'

print(f'Loading model from: {ckpt_path}')
model = FlowModel.load_from_checkpoint(ckpt_path)
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Load one batch of data
dataset = XCADataset(
    path='data/xca_full/train',
    crop_size=320,
    use_sauna_transform=True,
)
loader = DataLoader(dataset, batch_size=4, shuffle=True)
batch = next(iter(loader))

images = batch['image'].to(device)
geometry = batch.get('geometry', batch['label']).to(device)
labels = batch['label'].to(device)

print(f'\nBatch shapes:')
print(f'  images: {images.shape}')
print(f'  geometry: {geometry.shape}')
print(f'  labels: {labels.shape}')

# Simulate forward pass
model.train()
noise = torch.randn_like(geometry)

# Flow matching
t, xt, ut = model.flow_matcher.sample_location_and_conditional_flow(noise, geometry)

# Forward through UNet
with torch.enable_grad():
    # Get model predictions
    v = model.unet(xt, t, images)
    
    # Compute individual losses
    loss_inputs = {
        'v': v,
        'ut': ut,
        'xt': xt,
        'geometry': geometry,
        't': t,
        'hard_labels': labels,
        'x0': noise,
    }
    
    # Get loss function
    import inspect
    sig = inspect.signature(model.loss_fn.forward)
    filtered = {k: v for k, v in loss_inputs.items() if k in sig.parameters}
    
    total_loss, loss_dict = model.loss_fn(**filtered)
    
    print(f'\n📊 Individual Losses:')
    for name, value in loss_dict.items():
        print(f'  {name}_loss: {value.item():.6f}')
    print(f'  total_loss: {total_loss.item():.6f}')
    
    # Compute gradients for each loss separately
    print(f'\n🔍 Gradient Conflict Analysis:')
    
    # Get a sample parameter to check gradients
    sample_param = None
    for name, param in model.unet.named_parameters():
        if param.requires_grad and 'weight' in name:
            sample_param = param
            param_name = name
            break
    
    if sample_param is not None:
        print(f'  Analyzing parameter: {param_name[:50]}...')
        print(f'  Parameter shape: {sample_param.shape}')
        
        # Compute gradient for flow loss
        model.zero_grad()
        flow_loss = loss_dict.get('flow', loss_dict.get('l1', None))
        if flow_loss is not None:
            flow_loss.backward(retain_graph=True)
            flow_grad = sample_param.grad.clone()
            flow_grad_norm = flow_grad.norm().item()
            print(f'\n  Flow loss gradient norm: {flow_grad_norm:.6f}')
        
        # Compute gradient for geometry losses
        model.zero_grad()
        bce_loss = loss_dict.get('bce', None)
        dice_loss = loss_dict.get('dice', None)
        
        if bce_loss is not None and dice_loss is not None:
            geo_loss = bce_loss + dice_loss
            geo_loss.backward(retain_graph=True)
            geo_grad = sample_param.grad.clone()
            geo_grad_norm = geo_grad.norm().item()
            print(f'  Geometry loss gradient norm: {geo_grad_norm:.6f}')
            
            # Compute cosine similarity (conflict indicator)
            cos_sim = F.cosine_similarity(
                flow_grad.flatten().unsqueeze(0),
                geo_grad.flatten().unsqueeze(0)
            ).item()
            
            print(f'\n  📐 Gradient Cosine Similarity: {cos_sim:.4f}')
            if cos_sim < 0:
                print(f'  ❌ CONFLICT DETECTED! (negative cosine similarity)')
                print(f'     Gradients are pointing in opposite directions')
            elif cos_sim < 0.5:
                print(f'  ⚠️  Weak alignment (cosine sim < 0.5)')
                print(f'     Some conflict may exist')
            else:
                print(f'  ✅ Good alignment (cosine sim >= 0.5)')
                print(f'     No significant conflict')
            
            # Compute gradient magnitude ratio
            ratio = geo_grad_norm / (flow_grad_norm + 1e-8)
            print(f'\n  📊 Gradient Magnitude Ratio (geo/flow): {ratio:.4f}')
            if ratio > 2.0:
                print(f'  ⚠️  Geometry gradients are much larger (>{ratio:.1f}x)')
            elif ratio < 0.5:
                print(f'  ⚠️  Flow gradients are much larger (>{1/ratio:.1f}x)')
            else:
                print(f'  ✅ Gradients are balanced')
        else:
            print('  ⚠️  Could not compute geometry gradients')
    
    print(f'\n💡 Interpretation:')
    print(f'  - Cosine similarity: measures gradient direction alignment')
    print(f'    * 1.0 = perfect alignment (no conflict)')
    print(f'    * 0.0 = orthogonal (independent)')
    print(f'    * -1.0 = opposite directions (maximum conflict)')
    print(f'  - Gradient ratio: measures relative gradient magnitudes')
    print(f'    * ~1.0 = balanced contributions')
    print(f'    * >2.0 or <0.5 = one loss dominates')
