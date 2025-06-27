import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, List, Dict, Any
from enum import Enum


class DepthwiseSeparableLinear(nn.Module):
    
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.pointwise = nn.Linear(in_features, out_features, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(x)


class DepthwiseSeparableConv2d(nn.Module):    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3, 
        stride: int = 1, 
        padding: int = 1,
        bias: bool = False
    ):
        super().__init__()
        
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, 
            kernel_size=kernel_size, stride=stride, padding=padding,
            groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class SS2D(nn.Module):    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        dropout: float = 0.0,
        conv_bias: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        
        self.in_proj = DepthwiseSeparableLinear(self.d_model, self.d_inner * 2, bias=bias)
        
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
        )
        
        self.act = nn.SiLU()
        
        self.mamba = nn.Sequential(
            DepthwiseSeparableLinear(self.d_inner, self.d_inner),
            nn.SiLU(),
            DepthwiseSeparableLinear(self.d_inner, self.d_inner)
        )
        
        self.norm = nn.LayerNorm(self.d_inner)
        self.out_proj = DepthwiseSeparableLinear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.dt_bias = nn.Parameter(torch.zeros(self.d_inner))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        
        xz = self.in_proj(x)
        x_part, z = xz.chunk(2, dim=-1)
        
        x_part = x_part.permute(0, 3, 1, 2)
        x_part = self.act(self.conv2d(x_part))
        
        x_seq = x_part.flatten(2).permute(0, 2, 1)
        x_seq = self.mamba(x_seq + self.dt_bias)
        
        x_seq = x_seq.permute(0, 2, 1).reshape(B, -1, H, W)
        x_seq = x_seq.permute(0, 2, 3, 1)
        
        x_norm = self.norm(x_seq)
        x_norm = x_norm * F.silu(z)
        
        out = self.out_proj(x_norm)
        out = self.dropout(out)
        
        return out


class VSSBlock(nn.Module):    
    def __init__(self, dim: int, **kwargs):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.ss2d = SS2D(d_model=dim, **kwargs)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        x_ss2d = self.ss2d(x_norm)
        return x + x_ss2d


class DualVSSBlock(nn.Module):
    
    def __init__(self, dim: int, **kwargs):
        super().__init__()
        
        self.vss1 = VSSBlock(dim=dim, **kwargs)
        self.vss2 = VSSBlock(dim=dim, **kwargs)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vss1(x)
        x = self.vss2(x)
        return x


class SalientChannelAttentionModule(nn.Module):
    
    def __init__(self, channels: int, reduction_ratio: int = 16, bias: bool = False):
        super().__init__()
        
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        self.reduced_channels = max(1, channels // reduction_ratio)
        
        self.shared_mlp = nn.Sequential(
            DepthwiseSeparableLinear(channels, self.reduced_channels, bias=bias),
            nn.ReLU(inplace=True),
            DepthwiseSeparableLinear(self.reduced_channels, channels, bias=bias)
        )
        
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, features: torch.Tensor, saliency_map: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = features.shape
        
        if saliency_map.dim() == 3:
            saliency_map = saliency_map.unsqueeze(1)
        
        if saliency_map.shape[-2:] != features.shape[-2:]:
            saliency_map = F.interpolate(
                saliency_map, size=(height, width), 
                mode='bilinear', align_corners=False
            )
        
        saliency_filtered_features = features * saliency_map
        
        max_pooled = self.global_max_pool(saliency_filtered_features).view(batch_size, channels)
        avg_pooled = self.global_avg_pool(saliency_filtered_features).view(batch_size, channels)
        
        max_out = self.shared_mlp(max_pooled)
        avg_out = self.shared_mlp(avg_pooled)
        
        combined_out = max_out + avg_out
        channel_attention = self.sigmoid(combined_out).view(batch_size, channels, 1, 1)
        
        return channel_attention


class SeparableSpatialAttentionModule(nn.Module):
    
    def __init__(
        self,
        channels: int,
        kernel_sizes: Tuple[int, ...] = (1, 3, 5),
        dilation_rates: Tuple[int, ...] = (4, 8, 16),
        bias: bool = False
    ):
        super().__init__()
        
        self.channels = channels
        self.kernel_sizes = kernel_sizes
        self.dilation_rates = dilation_rates
        
        self.multi_scale_convs = nn.ModuleList()
        for kernel_size in kernel_sizes:
            padding = kernel_size // 2
            conv = nn.Conv2d(
                channels, channels, kernel_size=kernel_size, padding=padding,
                groups=channels, bias=bias
            )
            self.multi_scale_convs.append(conv)
        
        self.dilated_convs = nn.ModuleList()
        for dilation_rate in dilation_rates:
            padding = dilation_rate
            conv = nn.Conv2d(
                channels, channels, kernel_size=3, padding=padding,
                dilation=dilation_rate, groups=channels, bias=bias
            )
            self.dilated_convs.append(conv)
        
        total_multi_scale_channels = len(kernel_sizes) * channels
        total_dilated_channels = len(dilation_rates) * channels
        
        self.multi_scale_fusion = DepthwiseSeparableConv2d(
            total_multi_scale_channels, channels, kernel_size=1, padding=0, bias=bias
        )
        
        self.dilated_fusion = DepthwiseSeparableConv2d(
            total_dilated_channels, channels, kernel_size=1, padding=0, bias=bias
        )
        
        self.spatial_conv = DepthwiseSeparableConv2d(
            channels, 1, kernel_size=1, padding=0, bias=bias
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        multi_scale_features = []
        for conv in self.multi_scale_convs:
            conv_out = conv(features)
            multi_scale_features.append(conv_out)
        
        multi_scale_concat = torch.cat(multi_scale_features, dim=1)
        multi_scale_fused = self.multi_scale_fusion(multi_scale_concat)
        
        dilated_features = []
        for conv in self.dilated_convs:
            conv_out = conv(features)
            dilated_features.append(conv_out)
        
        dilated_concat = torch.cat(dilated_features, dim=1)
        dilated_fused = self.dilated_fusion(dilated_concat)
        
        combined_features = multi_scale_fused + dilated_fused
        spatial_attention = self.spatial_conv(combined_features)
        spatial_attention = self.sigmoid(spatial_attention)
        
        return spatial_attention


class ASRAMAttention(nn.Module):    
    def __init__(self, channels: int, reduction_ratio: int = 16, **kwargs):
        super().__init__()
        
        self.channels = channels
        self.scam = SalientChannelAttentionModule(channels=channels, reduction_ratio=reduction_ratio)
        self.ssam = SeparableSpatialAttentionModule(channels=channels, **kwargs)
        
    def forward(self, features: torch.Tensor, saliency_map: torch.Tensor) -> torch.Tensor:
        original_features = features
        
        channel_attention = self.scam(features, saliency_map)
        channel_refined_features = features * channel_attention
        
        spatial_attention = self.ssam(features)
        spatially_refined_features = channel_refined_features * spatial_attention
        
        output_features = original_features + spatially_refined_features
        
        return output_features


class ScalingOperation(Enum):
    MERGE = "merge"     
    EXPAND = "expand"   


class PatchMerging(nn.Module):
    
    def __init__(self, dim: int, dim_out: Optional[int] = None):
        super().__init__()
        dim_out = dim_out or (2 * dim)
        self.dim = dim
        self.dim_out = dim_out
        
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = DepthwiseSeparableLinear(4 * dim, dim_out, bias=False)
        if hasattr(self.reduction.pointwise, 'weight'):
            nn.init.xavier_uniform_(self.reduction.pointwise.weight)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        
        if H % 2 != 0 or W % 2 != 0:
            pad_h = H % 2
            pad_w = W % 2
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            _, H, W, _ = x.shape
        
        x0 = x[:, 0::2, 0::2, :]  # Top-left
        x1 = x[:, 1::2, 0::2, :]  # Bottom-left  
        x2 = x[:, 0::2, 1::2, :]  # Top-right
        x3 = x[:, 1::2, 1::2, :]  # Bottom-right
        
        x_merged = torch.cat([x0, x1, x2, x3], dim=-1)
        x_merged = self.norm(x_merged)
        x_merged = self.reduction(x_merged)
        
        return x_merged


class PatchExpanding(nn.Module):
    
    def __init__(self, dim: int, dim_out: Optional[int] = None):
        super().__init__()
        dim_out = dim_out or (dim // 2)
        self.dim = dim
        self.dim_out = dim_out
        self.expand_dim = 4 * dim_out
        
        self.norm = nn.LayerNorm(dim)
        self.expansion = DepthwiseSeparableLinear(dim, self.expand_dim, bias=False)

        if hasattr(self.expansion.pointwise, 'weight'):
            nn.init.xavier_uniform_(self.expansion.pointwise.weight)
            with torch.no_grad():
                self.expansion.pointwise.weight.mul_(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        
        x = self.norm(x)
        x = self.expansion(x)
        
        x = x.view(B, H, W, 4, self.dim_out)
        x = x.permute(0, 1, 3, 2, 4)
        x = x.contiguous().view(B, H, 2, W, 2, self.dim_out)
        x = x.permute(0, 1, 3, 2, 4, 5)
        x = x.contiguous().view(B, 2*H, 2*W, self.dim_out)
        
        return x


class FocusVisionMambaBlock(nn.Module):    
    def __init__(
        self,
        dim: int,
        operation: ScalingOperation,
        dim_out: Optional[int] = None,
        vss_config: Optional[Dict[str, Any]] = None,
        asram_config: Optional[Dict[str, Any]] = None,
        use_checkpoint: bool = False
    ):
        super().__init__()
        
        self.dim = dim
        self.operation = operation
        self.use_checkpoint = use_checkpoint

        if dim_out is None:
            if operation == ScalingOperation.MERGE:
                dim_out = 2 * dim  
            else:  
                dim_out = dim // 2  
        self.dim_out = dim_out
        self.vss_config = self._setup_vss_config(vss_config)
        self.asram_config = self._setup_asram_config(asram_config)

        self._build_architecture()
        
    def _setup_vss_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        defaults = {
            'd_state': 16,
            'd_conv': 3,
            'expand': 2,
            'dropout': 0.0,
            'conv_bias': True,
            'bias': False
        }
        return {**defaults, **(config or {})}
        
    def _setup_asram_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        defaults = {
            'reduction_ratio': 16,
            'kernel_sizes': (1, 3, 5),
            'dilation_rates': (4, 8, 16),
            'bias': False
        }
        return {**defaults, **(config or {})}
        
    def _build_architecture(self) -> None:
        if self.operation == ScalingOperation.MERGE:
            self.patch_op = PatchMerging(self.dim, self.dim_out)
            self.processing_dim = self.dim_out
        else:  
            self.patch_op = PatchExpanding(self.dim, self.dim_out)
            self.processing_dim = self.dim_out

        self.vss_blocks = DualVSSBlock(
            dim=self.processing_dim,
            **self.vss_config
        )

        self.asram = ASRAMAttention(
            channels=self.processing_dim,
            **self.asram_config
        )

        self.norm = nn.LayerNorm(self.processing_dim)
        
    def _apply_gradient_checkpointing(self, func: callable, *args, **kwargs) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(func, *args, **kwargs)
        return func(*args, **kwargs)
        
    def forward(self, x: torch.Tensor, saliency_map: torch.Tensor) -> torch.Tensor:
        x_scaled = self.patch_op(x) 
        B, H_new, W_new, C_new = x_scaled.shape
        saliency_adapted = self._adapt_saliency_map(saliency_map, (H_new, W_new))
        x_vss = self._apply_gradient_checkpointing(self.vss_blocks, x_scaled)
        x_asram = x_vss.permute(0, 3, 1, 2)  
        x_attended = self._apply_gradient_checkpointing(
            self.asram, x_asram, saliency_adapted
        )

        x_out = x_attended.permute(0, 2, 3, 1)  
        x_out = self.norm(x_out)
        
        return x_out
        
    def _adapt_saliency_map(
        self, 
        saliency_map: torch.Tensor, 
        target_size: Tuple[int, int]
    ) -> torch.Tensor:
        if saliency_map.dim() == 3:
            saliency_map = saliency_map.unsqueeze(1) 
            
        current_size = saliency_map.shape[-2:]
        if current_size != target_size:
            saliency_map = F.interpolate(
                saliency_map,
                size=target_size,
                mode='bilinear',
                align_corners=False
            )
            
        return saliency_map


class ResidualConvBlock(nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_residual: bool = True
    ):
        super().__init__()
        
        self.use_residual = use_residual and (in_channels == out_channels) and (stride == 1)

        self.conv1 = DepthwiseSeparableConv2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU(inplace=True)
        
        self.conv2 = DepthwiseSeparableConv2d(
            out_channels, out_channels, kernel_size, 1, padding
        )
        self.norm2 = nn.BatchNorm2d(out_channels)
        if not self.use_residual and in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
            self.residual_norm = nn.BatchNorm2d(out_channels)
        
        self.act2 = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        
        out = self.conv2(out)
        out = self.norm2(out)

        if self.use_residual:
            out += identity
        elif hasattr(self, 'residual_conv'):
            identity = self.residual_conv(identity)
            identity = self.residual_norm(identity)
            out += identity
        
        out = self.act2(out)
        
        return out


class SkipConnectionFusion(nn.Module):    
    def __init__(self, encoder_dim: int, decoder_dim: int):
        super().__init__()
        if encoder_dim != decoder_dim:
            self.align = DepthwiseSeparableConv2d(encoder_dim, decoder_dim, kernel_size=1, padding=0)
        else:
            self.align = nn.Identity()
        self.fusion = nn.Sequential(
            nn.LayerNorm(decoder_dim),
            nn.GELU()
        )
        
    def forward(self, encoder_feat: torch.Tensor, decoder_feat: torch.Tensor) -> torch.Tensor:
        decoder_feat = decoder_feat.permute(0, 3, 1, 2)  
        encoder_feat = self.align(encoder_feat)
        if encoder_feat.shape[-2:] != decoder_feat.shape[-2:]:
            encoder_feat = F.interpolate(
                encoder_feat, size=decoder_feat.shape[-2:],
                mode='bilinear', align_corners=False
            )

        fused = encoder_feat + decoder_feat  

        fused = fused.permute(0, 2, 3, 1) 
        fused = self.fusion(fused)
        
        return fused



class MixLVMMExpertNetwork(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
        use_checkpoint: bool = False
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.use_checkpoint = use_checkpoint

        self.encoder_dims = [24, 48, 96, 192]  
        self.decoder_dims = self.encoder_dims[::-1]  
        self.initial_conv = ResidualConvBlock(
            in_channels=in_channels,
            out_channels=24, 
            stride=1,
            use_residual=False
        )

        self.encoder_blocks = nn.ModuleList()

        encoder_configs = [
            (24, 24),   
            (24, 48),   
            (48, 96),   
            (96, 192),  
        ]
        
        for i, (dim_in, dim_out) in enumerate(encoder_configs):
            block = FocusVisionMambaBlock(
                dim=dim_in,
                operation=ScalingOperation.MERGE,
                dim_out=dim_out,
                use_checkpoint=use_checkpoint
            )
            self.encoder_blocks.append(block)

        bridge_dim = self.encoder_dims[-1]  
        self.bridge = ASRAMAttention(
            channels=bridge_dim,
            reduction_ratio=16
        )
        self.bridge_norm = nn.LayerNorm(bridge_dim)

        self.decoder_blocks = nn.ModuleList()
        self.skip_fusions = nn.ModuleList()

        decoder_configs = [
            (192, 96), 
            (96, 48),   
            (48, 24),   
            (24, 24),  
        ]

        skip_encoder_dims = [192, 96, 48, 24]  
        
        for i, ((dim_in, dim_out), encoder_dim) in enumerate(zip(decoder_configs, skip_encoder_dims)):
            block = FocusVisionMambaBlock(
                dim=dim_in,
                operation=ScalingOperation.EXPAND,
                dim_out=dim_out,
                use_checkpoint=use_checkpoint
            )
            self.decoder_blocks.append(block)
            fusion = SkipConnectionFusion(encoder_dim, dim_out)
            self.skip_fusions.append(fusion)

        self.final_conv = ResidualConvBlock(
            in_channels=24,  
            out_channels=out_channels,
            use_residual=False
        )

        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(
        self, 
        rgb_image: torch.Tensor, 
        saliency_map: torch.Tensor
    ) -> torch.Tensor:
        x = self.initial_conv(rgb_image) 
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  

        encoder_features = []

        for i, encoder_block in enumerate(self.encoder_blocks):
            x = encoder_block(x, saliency_map) 

            encoder_feat = x.permute(0, 3, 1, 2)  
            encoder_features.append(encoder_feat)

        x_bridge = x.permute(0, 3, 1, 2)  
        x_bridge = self.bridge(x_bridge, saliency_map)
        x = x_bridge.permute(0, 2, 3, 1) 
        x = self.bridge_norm(x)

        for i, (decoder_block, skip_fusion) in enumerate(zip(self.decoder_blocks, self.skip_fusions)):
            encoder_level = len(encoder_features) - 1 - i
            encoder_feat = encoder_features[encoder_level]

            x = decoder_block(x, saliency_map)
            x = skip_fusion(encoder_feat, x)
        x_out = x.permute(0, 3, 1, 2) 
        mask = self.final_conv(x_out) 
        
        return mask