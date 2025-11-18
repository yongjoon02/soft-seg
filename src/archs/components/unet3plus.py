"""UNet3Plus: Full-scale deep supervision for medical image segmentation.

Refactored version with shared components to reduce code duplication.
Maintains exact same functionality as original implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def init_weights(module, init_type='kaiming'):
    """Initialize weights for Conv2d and BatchNorm2d layers."""
    if isinstance(module, nn.Conv2d):
        if init_type == 'kaiming':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


class unetConv2(nn.Module):
    """Double convolution block with optional batch normalization."""
    
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        
        for i in range(1, n + 1):
            if is_batchnorm:
                conv = nn.Sequential(
                    nn.Conv2d(in_size, out_size, ks, stride, padding),
                    nn.BatchNorm2d(out_size),
                    nn.ReLU(inplace=True)
                )
            else:
                conv = nn.Sequential(
                    nn.Conv2d(in_size, out_size, ks, stride, padding),
                    nn.ReLU(inplace=True)
                )
            setattr(self, 'conv%d' % i, conv)
            in_size = out_size

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)
        return x


class UNet3PlusEncoder(nn.Module):
    """Shared encoder for all UNet3Plus variants."""
    
    def __init__(self, in_channels, filters, is_batchnorm):
        super().__init__()
        self.conv1 = unetConv2(in_channels, filters[0], is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = unetConv2(filters[0], filters[1], is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = unetConv2(filters[1], filters[2], is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = unetConv2(filters[2], filters[3], is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        self.conv5 = unetConv2(filters[3], filters[4], is_batchnorm)
    
    def forward(self, x):
        h1 = self.conv1(x)
        h2 = self.conv2(self.maxpool1(h1))
        h3 = self.conv3(self.maxpool2(h2))
        h4 = self.conv4(self.maxpool3(h3))
        h5 = self.conv5(self.maxpool4(h4))
        return h1, h2, h3, h4, h5


class SkipConnectionBlock(nn.Module):
    """Single skip connection block for full-scale feature fusion."""
    
    def __init__(self, in_channels, cat_channels, scale_factor=None, mode='cat'):
        """
        Args:
            in_channels: Input channel size
            cat_channels: Output channel size (after projection)
            scale_factor: For pooling (if mode='pool') or upsampling (if mode='upsample')
            mode: 'pool', 'cat', or 'upsample'
        """
        super().__init__()
        self.mode = mode
        
        if mode == 'pool':
            self.resize = nn.MaxPool2d(scale_factor, scale_factor, ceil_mode=True)
        elif mode == 'upsample':
            self.resize = nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        else:  # mode == 'cat'
            self.resize = nn.Identity()
        
        self.conv = nn.Conv2d(in_channels, cat_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(cat_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.resize(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DecoderStage(nn.Module):
    """Single decoder stage that fuses features from all encoder levels."""
    
    def __init__(self, filters, cat_channels, up_channels, stage_idx):
        """
        Args:
            filters: List of filter sizes [64, 128, 256, 512, 1024]
            cat_channels: Channel size for each skip connection (usually filters[0])
            up_channels: Total channels after concatenation (cat_channels * 5)
            stage_idx: Which decoder stage (4, 3, 2, or 1)
        """
        super().__init__()
        self.stage_idx = stage_idx
        
        # Calculate scale factors for each encoder level
        # stage 4d: h1(8x), h2(4x), h3(2x), h4(1x), h5(2x up)
        # stage 3d: h1(4x), h2(2x), h3(1x), h4(2x up), h5(4x up)
        # stage 2d: h1(2x), h2(1x), h3(2x up), h4(4x up), h5(8x up)
        # stage 1d: h1(1x), h2(2x up), h3(4x up), h4(8x up), h5(16x up)
        
        scale_factors = {
            4: [8, 4, 2, None, 2],  # pool, pool, pool, cat, upsample
            3: [4, 2, None, 2, 4],
            2: [2, None, 2, 4, 8],
            1: [None, 2, 4, 8, 16]
        }
        
        modes = {
            4: ['pool', 'pool', 'pool', 'cat', 'upsample'],
            3: ['pool', 'pool', 'cat', 'upsample', 'upsample'],
            2: ['pool', 'cat', 'upsample', 'upsample', 'upsample'],
            1: ['cat', 'upsample', 'upsample', 'upsample', 'upsample']
        }
        
        scales = scale_factors[stage_idx]
        mode_list = modes[stage_idx]
        
        # Create skip connection blocks for each encoder level
        self.blocks = nn.ModuleList()
        for i, (scale, mode) in enumerate(zip(scales, mode_list)):
            if i < 4:  # h1, h2, h3, h4
                # For h4 position in stage 3d, 2d, 1d: use up_channels (prev_decoder)
                # For h3 position in stage 2d, 1d: use up_channels (prev_decoder)
                # For h2 position in stage 1d: use up_channels (prev_decoder)
                # Otherwise: use encoder feature channels
                if (i == 3 and stage_idx < 4) or (i == 2 and stage_idx < 3) or (i == 1 and stage_idx < 2):
                    in_ch = up_channels  # Using prev_decoder
                else:
                    in_ch = filters[i]  # Using encoder feature
            else:  # h5 position (always uses encoder h5, which is filters[4])
                in_ch = filters[4]
            
            self.blocks.append(SkipConnectionBlock(in_ch, cat_channels, scale, mode))
        
        # Fusion layer
        self.fusion_conv = nn.Conv2d(up_channels, up_channels, 3, padding=1)
        self.fusion_bn = nn.BatchNorm2d(up_channels)
        self.fusion_relu = nn.ReLU(inplace=True)
    
    def forward(self, encoder_features, decoder_outputs=None):
        """
        Args:
            encoder_features: Tuple of (h1, h2, h3, h4, h5) from encoder
            decoder_outputs: Dict of decoder outputs {hd4, hd3, hd2} or None
        
        Returns:
            Decoder output at this stage
        """
        h1, h2, h3, h4, h5 = encoder_features
        
        # Determine which features to use based on stage
        # Stage 4d: all encoder features
        # Stage 3d: h1, h2, h3 from encoder, hd4 from decoder, h5 from encoder
        # Stage 2d: h1, h2 from encoder, hd3, hd4 from decoder, h5 from encoder
        # Stage 1d: h1 from encoder, hd2, hd3, hd4 from decoder, h5 from encoder
        
        if self.stage_idx == 4:
            features = [h1, h2, h3, h4, h5]
        elif self.stage_idx == 3:
            features = [h1, h2, h3, decoder_outputs['hd4'], h5]
        elif self.stage_idx == 2:
            features = [h1, h2, decoder_outputs['hd3'], decoder_outputs['hd4'], h5]
        else:  # stage_idx == 1
            features = [h1, decoder_outputs['hd2'], decoder_outputs['hd3'], decoder_outputs['hd4'], h5]
        
        # Apply skip connection blocks
        processed = []
        for feat, block in zip(features, self.blocks):
            processed.append(block(feat))
        
        # Concatenate and fuse
        fused = torch.cat(processed, dim=1)
        output = self.fusion_relu(self.fusion_bn(self.fusion_conv(fused)))
        return output


class UNet3PlusBase(nn.Module):
    """Base class for UNet3Plus variants with shared encoder and decoder structure."""
    
    def __init__(self, in_channels=3, num_classes=2, feature_scale=4, 
                 is_deconv=True, is_batchnorm=True, use_deep_sup=False, use_cgm=False):
        super().__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.use_deep_sup = use_deep_sup
        self.use_cgm = use_cgm
        
        filters = [64, 128, 256, 512, 1024]
        
        # Shared encoder
        self.encoder = UNet3PlusEncoder(in_channels, filters, is_batchnorm)
        
        # Decoder configuration
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks
        
        # Decoder stages
        self.decoder_stage4 = DecoderStage(filters, self.CatChannels, self.UpChannels, 4)
        self.decoder_stage3 = DecoderStage(filters, self.CatChannels, self.UpChannels, 3)
        self.decoder_stage2 = DecoderStage(filters, self.CatChannels, self.UpChannels, 2)
        self.decoder_stage1 = DecoderStage(filters, self.CatChannels, self.UpChannels, 1)
        
        # Output convolutions
        self.outconv1 = nn.Conv2d(self.UpChannels, num_classes, 3, padding=1)
        self.outconv2 = nn.Conv2d(self.UpChannels, num_classes, 3, padding=1)
        self.outconv3 = nn.Conv2d(self.UpChannels, num_classes, 3, padding=1)
        self.outconv4 = nn.Conv2d(self.UpChannels, num_classes, 3, padding=1)
        self.outconv5 = nn.Conv2d(filters[4], num_classes, 3, padding=1)
        
        # Deep supervision upsampling (only for DeepSup variants)
        if use_deep_sup:
            self.upscore5 = nn.Upsample(scale_factor=16, mode='bilinear')
            self.upscore4 = nn.Upsample(scale_factor=8, mode='bilinear')
            self.upscore3 = nn.Upsample(scale_factor=4, mode='bilinear')
            self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')
        
        # CGM module (only for CGM variant)
        if use_cgm:
            self.cls = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Conv2d(filters[4], 2, 1),
                nn.AdaptiveMaxPool2d(1),
                nn.Sigmoid()
            )
    
    def forward_encoder(self, x):
        """Forward pass through encoder."""
        return self.encoder(x)
    
    def forward_decoder(self, encoder_features):
        """Forward pass through decoder stages."""
        h1, h2, h3, h4, h5 = encoder_features
        
        # Decoder stages (bottom-up)
        # Stage 4d: uses all encoder features
        hd4 = self.decoder_stage4((h1, h2, h3, h4, h5), decoder_outputs=None)
        
        # Stage 3d: uses hd4 for h4 position
        hd3 = self.decoder_stage3((h1, h2, h3, h4, h5), decoder_outputs={'hd4': hd4})
        
        # Stage 2d: uses hd3 and hd4
        hd2 = self.decoder_stage2((h1, h2, h3, h4, h5), decoder_outputs={'hd3': hd3, 'hd4': hd4})
        
        # Stage 1d: uses hd2, hd3, hd4
        hd1 = self.decoder_stage1((h1, h2, h3, h4, h5), decoder_outputs={'hd2': hd2, 'hd3': hd3, 'hd4': hd4})
        
        return hd1, hd2, hd3, hd4, h5
    
    def forward_outputs(self, hd1, hd2, hd3, hd4, h5):
        """Generate output predictions from decoder features."""
        d1 = self.outconv1(hd1)
        d2 = self.outconv2(hd2)
        d3 = self.outconv3(hd3)
        d4 = self.outconv4(hd4)
        d5 = self.outconv5(h5)
        return d1, d2, d3, d4, d5


class UNet3Plus(UNet3PlusBase):
    """UNet3Plus: Full-scale deep supervision (basic version)."""
    
    def __init__(self, in_channels=3, num_classes=2, feature_scale=4, 
                 is_deconv=True, is_batchnorm=True):
        super().__init__(in_channels, num_classes, feature_scale, 
                        is_deconv, is_batchnorm, use_deep_sup=False, use_cgm=False)
    
    def forward(self, x):
        # Encoder
        h1, h2, h3, h4, h5 = self.forward_encoder(x)
        
        # Decoder
        hd1, hd2, hd3, hd4, h5 = self.forward_decoder((h1, h2, h3, h4, h5))
        
        # Outputs
        d1, d2, d3, d4, d5 = self.forward_outputs(hd1, hd2, hd3, hd4, h5)
        
        # Return as dict (maintains compatibility)
        output = dict()
        aux_feature = [d2, d3, d4, d5]
        size = d1.size()[2:]
        aux_out = []
        for a in aux_feature:
            a = F.interpolate(a, size, mode='bilinear', align_corners=True)
            aux_out.append(a)
        output.update({"aux_out": aux_out})
        output["main_out"] = d1
        return output


class UNet_3Plus_DeepSup(UNet3PlusBase):
    """UNet3Plus with Deep Supervision."""
    
    def __init__(self, in_channels=3, num_classes=1, feature_scale=4, 
                 is_deconv=True, is_batchnorm=True):
        super().__init__(in_channels, num_classes, feature_scale, 
                        is_deconv, is_batchnorm, use_deep_sup=True, use_cgm=False)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')
    
    def forward(self, inputs):
        # Encoder
        h1, h2, h3, h4, h5 = self.forward_encoder(inputs)
        
        # Decoder
        hd1, hd2, hd3, hd4, h5 = self.forward_decoder((h1, h2, h3, h4, h5))
        
        # Outputs with upsampling for deep supervision
        d5 = self.outconv5(h5)
        d5 = self.upscore5(d5)
        
        d4 = self.outconv4(hd4)
        d4 = self.upscore4(d4)
        
        d3 = self.outconv3(hd3)
        d3 = self.upscore3(d3)
        
        d2 = self.outconv2(hd2)
        d2 = self.upscore2(d2)
        
        d1 = self.outconv1(hd1)
        
        # Return tuple with sigmoid (maintains compatibility)
        return F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5)


class UNet_3Plus_DeepSup_CGM(UNet3PlusBase):
    """UNet3Plus with Deep Supervision and Classification Guided Module."""
    
    def __init__(self, in_channels=3, num_classes=1, feature_scale=4, 
                 is_deconv=True, is_batchnorm=True, aux=True):
        super().__init__(in_channels, num_classes, feature_scale, 
                        is_deconv, is_batchnorm, use_deep_sup=True, use_cgm=True)
        self.aux = aux
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')
    
    def dotProduct(self, seg, cls):
        """Apply classification guidance to segmentation output."""
        B, N, H, W = seg.size()
        seg = seg.view(B, N, H * W)
        final = torch.einsum("ijk,ij->ijk", [seg, cls])
        final = final.view(B, N, H, W)
        return final
    
    def forward(self, inputs):
        # Encoder
        h1, h2, h3, h4, h5 = self.forward_encoder(inputs)
        
        # Classification branch
        cls_branch = self.cls(h5).squeeze(3).squeeze(2)  # (B,N,1,1)->(B,N)
        cls_branch_max = cls_branch.argmax(dim=1)
        cls_branch_max = cls_branch_max[:, np.newaxis].float()
        
        # Decoder
        hd1, hd2, hd3, hd4, h5 = self.forward_decoder((h1, h2, h3, h4, h5))
        
        # Outputs
        d1, d2, d3, d4, d5 = self.forward_outputs(hd1, hd2, hd3, hd4, h5)
        
        # Apply CGM
        size = d1.size()[2:]
        d1 = self.dotProduct(d1, cls_branch_max)
        d2 = self.dotProduct(d2, cls_branch_max)
        d3 = self.dotProduct(d3, cls_branch_max)
        d4 = self.dotProduct(d4, cls_branch_max)
        d5 = self.dotProduct(d5, cls_branch_max)
        
        # Return tuple with sigmoid (maintains compatibility)
        return F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5)


if __name__ == "__main__":
    net = UNet3Plus(in_channels=1, num_classes=2)
    inputs = torch.randn(1, 1, 128, 128)
    outputs = net(inputs)
    if isinstance(outputs, dict):
        print({k: v.shape if isinstance(v, torch.Tensor) else [t.shape for t in v] 
               for k, v in outputs.items()})
    else:
        print(outputs.shape if isinstance(outputs, torch.Tensor) else [t.shape for t in outputs])
