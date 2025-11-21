"""DSCNet: Dynamic Snake Convolution Network for vessel segmentation.

Based on official implementation: https://github.com/YaoleiQi/DSCNet
Paper: "DSCNet: Dynamic Snake Convolution based on Topological Geometric Constraints for Tubular Structure Segmentation"

Note: This implementation requires the DSConv module. If S3_DSConv_pro is not available,
a simplified approximation is used.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Configuration: Use official DSConv or approximation
USE_OFFICIAL_DSCONV = True  # Set to True to use DSConv_pro, False for approximation

if USE_OFFICIAL_DSCONV:
    try:
        from .S3_DSConv_pro import DSConv_pro
        HAS_DSCONV = True
    except ImportError:
        HAS_DSCONV = False
        print("⚠ DSConv_pro not found. Using approximation.")
else:
    HAS_DSCONV = False


class DSConvApprox(nn.Module):
    """Approximated Dynamic Snake Convolution (lightweight version).
    
    This is a simplified version that captures the essence of snake convolution
    without the computational overhead of deformable convolution.
    
    Uses directional convolutions with snake-like activation patterns.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 9,
                 extend_scope: float = 1.0, morph: int = 0, if_offset: bool = True, device: str = 'cpu'):
        super().__init__()
        self.morph = morph  # 0: x-axis, 1: y-axis
        
        # Use asymmetric kernels to capture directional features
        if morph == 0:  # x-axis: horizontal structures
            # Wide horizontal kernel
            kernel_h, kernel_w = 3, 7
        else:  # y-axis: vertical structures
            # Tall vertical kernel
            kernel_h, kernel_w = 7, 3
        
        padding_h = kernel_h // 2
        padding_w = kernel_w // 2
        
        self.conv = nn.Conv2d(in_channels, out_channels, (kernel_h, kernel_w), 
                             padding=(padding_h, padding_w))
        self.gn = nn.GroupNorm(out_channels // 4, out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Snake-like activation enhancement (optional)
        if if_offset:
            self.alpha = nn.Parameter(torch.ones(1, out_channels, 1, 1) * 0.1)
        else:
            self.alpha = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.gn(out)
        
        # Add snake-like modulation
        if self.alpha is not None:
            out = out + self.alpha * torch.sin(out)
        
        return self.relu(out)


class EncoderConv(nn.Module):
    """Encoder convolution block with GroupNorm."""
    
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.gn(self.conv(x)))


class DecoderConv(nn.Module):
    """Decoder convolution block with GroupNorm."""
    
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.gn(self.conv(x)))


class DSCNet(nn.Module):
    """Dynamic Snake Convolution Network for tubular structure segmentation.
    
    Official architecture from DSCNet paper. Uses Dynamic Snake Convolution (DSConv)
    in x and y directions along with standard convolution for multi-directional feature extraction.
    
    Args:
        in_channels: Number of input channels (default: 1 for grayscale)
        num_classes: Number of output classes (default: 2 for binary segmentation)
        base_channels: Base number of channels (default: 32)
        kernel_size: Kernel size for DSConv (default: 9)
        extend_scope: Range to expand for DSConv (default: 1.0)
        if_offset: Whether deformation is required (default: True)
        device: Device to use (default: 'cpu')
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_channels: int = 32,
        kernel_size: int = 9,
        extend_scope: float = 1.0,
        if_offset: bool = True,
        device: str = 'cpu',
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.extend_scope = extend_scope
        self.if_offset = if_offset
        self.relu = nn.ReLU(inplace=True)
        self.number = base_channels
        
        # Select DSConv implementation
        if HAS_DSCONV:
            DSConv = DSConv_pro
        else:
            DSConv = DSConvApprox
        
        # Block 0: Input layer (3 parallel paths: standard, x-axis, y-axis)
        self.conv00 = EncoderConv(in_channels, self.number)
        self.conv0x = DSConv(in_channels, self.number, kernel_size, extend_scope, 0, if_offset, device)
        self.conv0y = DSConv(in_channels, self.number, kernel_size, extend_scope, 1, if_offset, device)
        self.conv1 = EncoderConv(3 * self.number, self.number)

        # Block 1: First downsampling
        self.conv20 = EncoderConv(self.number, 2 * self.number)
        self.conv2x = DSConv(self.number, 2 * self.number, kernel_size, extend_scope, 0, if_offset, device)
        self.conv2y = DSConv(self.number, 2 * self.number, kernel_size, extend_scope, 1, if_offset, device)
        self.conv3 = EncoderConv(6 * self.number, 2 * self.number)

        # Block 2: Second downsampling
        self.conv40 = EncoderConv(2 * self.number, 4 * self.number)
        self.conv4x = DSConv(2 * self.number, 4 * self.number, kernel_size, extend_scope, 0, if_offset, device)
        self.conv4y = DSConv(2 * self.number, 4 * self.number, kernel_size, extend_scope, 1, if_offset, device)
        self.conv5 = EncoderConv(12 * self.number, 4 * self.number)

        # Block 3: Third downsampling (bottleneck)
        self.conv60 = EncoderConv(4 * self.number, 8 * self.number)
        self.conv6x = DSConv(4 * self.number, 8 * self.number, kernel_size, extend_scope, 0, if_offset, device)
        self.conv6y = DSConv(4 * self.number, 8 * self.number, kernel_size, extend_scope, 1, if_offset, device)
        self.conv7 = EncoderConv(24 * self.number, 8 * self.number)

        # Block 4: First upsampling
        self.conv120 = EncoderConv(12 * self.number, 4 * self.number)
        self.conv12x = DSConv(12 * self.number, 4 * self.number, kernel_size, extend_scope, 0, if_offset, device)
        self.conv12y = DSConv(12 * self.number, 4 * self.number, kernel_size, extend_scope, 1, if_offset, device)
        self.conv13 = EncoderConv(12 * self.number, 4 * self.number)

        # Block 5: Second upsampling
        self.conv140 = DecoderConv(6 * self.number, 2 * self.number)
        self.conv14x = DSConv(6 * self.number, 2 * self.number, kernel_size, extend_scope, 0, if_offset, device)
        self.conv14y = DSConv(6 * self.number, 2 * self.number, kernel_size, extend_scope, 1, if_offset, device)
        self.conv15 = DecoderConv(6 * self.number, 2 * self.number)

        # Block 6: Third upsampling (output)
        self.conv160 = DecoderConv(3 * self.number, self.number)
        self.conv16x = DSConv(3 * self.number, self.number, kernel_size, extend_scope, 0, if_offset, device)
        self.conv16y = DSConv(3 * self.number, self.number, kernel_size, extend_scope, 1, if_offset, device)
        self.conv17 = DecoderConv(3 * self.number, self.number)

        # Output layer
        self.out_conv = nn.Conv2d(self.number, num_classes, 1)
        
        # Pooling and upsampling
        self.maxpooling = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        
        # Note: Sigmoid removed - will be applied in loss function
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass following official DSCNet architecture.
        
        Each block uses 3 parallel paths:
        - Standard convolution (conv*0)
        - DSConv in x-axis direction (conv*x)
        - DSConv in y-axis direction (conv*y)
        These are concatenated and fused.
        """
        # Block 0: Input processing
        x_00_0 = self.conv00(x)
        x_0x_0 = self.conv0x(x)
        x_0y_0 = self.conv0y(x)
        x_0_1 = self.conv1(torch.cat([x_00_0, x_0x_0, x_0y_0], dim=1))

        # Block 1: First encoder stage
        x = self.maxpooling(x_0_1)
        x_20_0 = self.conv20(x)
        x_2x_0 = self.conv2x(x)
        x_2y_0 = self.conv2y(x)
        x_1_1 = self.conv3(torch.cat([x_20_0, x_2x_0, x_2y_0], dim=1))

        # Block 2: Second encoder stage
        x = self.maxpooling(x_1_1)
        x_40_0 = self.conv40(x)
        x_4x_0 = self.conv4x(x)
        x_4y_0 = self.conv4y(x)
        x_2_1 = self.conv5(torch.cat([x_40_0, x_4x_0, x_4y_0], dim=1))

        # Block 3: Bottleneck
        x = self.maxpooling(x_2_1)
        x_60_0 = self.conv60(x)
        x_6x_0 = self.conv6x(x)
        x_6y_0 = self.conv6y(x)
        x_3_1 = self.conv7(torch.cat([x_60_0, x_6x_0, x_6y_0], dim=1))

        # Block 4: First decoder stage with skip connection
        x = self.up(x_3_1)
        x_concat = torch.cat([x, x_2_1], dim=1)
        x_120_2 = self.conv120(x_concat)
        x_12x_2 = self.conv12x(x_concat)
        x_12y_2 = self.conv12y(x_concat)
        x_2_3 = self.conv13(torch.cat([x_120_2, x_12x_2, x_12y_2], dim=1))

        # Block 5: Second decoder stage with skip connection
        x = self.up(x_2_3)
        x_concat = torch.cat([x, x_1_1], dim=1)
        x_140_2 = self.conv140(x_concat)
        x_14x_2 = self.conv14x(x_concat)
        x_14y_2 = self.conv14y(x_concat)
        x_1_3 = self.conv15(torch.cat([x_140_2, x_14x_2, x_14y_2], dim=1))

        # Block 6: Third decoder stage with skip connection
        x = self.up(x_1_3)
        x_concat = torch.cat([x, x_0_1], dim=1)
        x_160_2 = self.conv160(x_concat)
        x_16x_2 = self.conv16x(x_concat)
        x_16y_2 = self.conv16y(x_concat)
        x_0_3 = self.conv17(torch.cat([x_160_2, x_16x_2, x_16y_2], dim=1))
        
        # Output (no sigmoid - will be applied in loss)
        out = self.out_conv(x_0_3)
        
        return out


if __name__ == "__main__":
    # Test the model
    model = DSCNet(in_channels=1, num_classes=2, base_channels=32)
    x = torch.randn(2, 1, 224, 224)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

