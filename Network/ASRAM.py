import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union
import math


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
        dilation: int = 1,
        bias: bool = False
    ):
        super().__init__()

        self.depthwise = nn.Conv2d(
            in_channels, 
            in_channels, 
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False
        )

        self.pointwise = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=1,
            bias=bias
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class SalientChannelAttentionModule(nn.Module):
    def __init__(
        self,
        channels: int,
        reduction_ratio: int = 16,
        bias: bool = False
    ):
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
        
    def forward(
        self, 
        features: torch.Tensor, 
        saliency_map: torch.Tensor
    ) -> torch.Tensor:
 
        batch_size, channels, height, width = features.shape

        if saliency_map.dim() == 3:  
            saliency_map = saliency_map.unsqueeze(1) 

        if saliency_map.shape[-2:] != features.shape[-2:]:
            saliency_map = F.interpolate(
                saliency_map, 
                size=(height, width), 
                mode='bilinear', 
                align_corners=False
            )

        saliency_filtered_features = features * saliency_map 

        max_pooled = self.global_max_pool(saliency_filtered_features) 
        max_pooled = max_pooled.view(batch_size, channels)  
        
        avg_pooled = self.global_avg_pool(saliency_filtered_features) 
        avg_pooled = avg_pooled.view(batch_size, channels) 

        max_out = self.shared_mlp(max_pooled) 
        avg_out = self.shared_mlp(avg_pooled)
        
        combined_out = max_out + avg_out  
        channel_attention = self.sigmoid(combined_out) 

        channel_attention = channel_attention.view(batch_size, channels, 1, 1) 
        
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
                channels, 
                channels, 
                kernel_size=kernel_size, 
                padding=padding,
                groups=channels,  
                bias=bias
            )
            self.multi_scale_convs.append(conv)

        self.dilated_convs = nn.ModuleList()
        for dilation_rate in dilation_rates:
            padding = dilation_rate
            conv = nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=padding,
                dilation=dilation_rate,
                groups=channels, 
                bias=bias
            )
            self.dilated_convs.append(conv)

        total_multi_scale_channels = len(kernel_sizes) * channels
        total_dilated_channels = len(dilation_rates) * channels

        self.multi_scale_fusion = DepthwiseSeparableConv2d(
            total_multi_scale_channels, 
            channels, 
            kernel_size=1,
            padding=0,
            bias=bias
        )
        
        self.dilated_fusion = DepthwiseSeparableConv2d(
            total_dilated_channels, 
            channels, 
            kernel_size=1,
            padding=0,
            bias=bias
        )

        self.spatial_conv = DepthwiseSeparableConv2d(
            channels, 
            1, 
            kernel_size=1,
            padding=0,
            bias=bias
        )

        self.sigmoid = nn.Sigmoid()
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:

        batch_size, channels, height, width = features.shape
        
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
    
    def __init__(
        self,
        channels: int,
        reduction_ratio: int = 16,
        kernel_sizes: Tuple[int, ...] = (1, 3, 5),
        dilation_rates: Tuple[int, ...] = (4, 8, 16),
        bias: bool = False
    ):
        super().__init__()
        
        self.channels = channels

        self.scam = SalientChannelAttentionModule(
            channels=channels,
            reduction_ratio=reduction_ratio,
            bias=bias
        )

        self.ssam = SeparableSpatialAttentionModule(
            channels=channels,
            kernel_sizes=kernel_sizes,
            dilation_rates=dilation_rates,
            bias=bias
        )
        
    def forward(
        self, 
        features: torch.Tensor, 
        saliency_map: torch.Tensor
    ) -> torch.Tensor:

        original_features = features

        channel_attention = self.scam(features, saliency_map)  

        channel_refined_features = features * channel_attention  

        spatial_attention = self.ssam(features)  

        spatially_refined_features = channel_refined_features * spatial_attention 

        output_features = original_features + spatially_refined_features 
        
        return output_features


class ASRAMBridge(nn.Module):
    
    def __init__(
        self,
        channels: int,
        reduction_ratio: int = 16,
        use_norm: bool = True,
        use_activation: bool = False
    ):
        super().__init__()

        self.asram = ASRAMAttention(
            channels=channels,
            reduction_ratio=reduction_ratio
        )

        self.norm = nn.LayerNorm(channels) if use_norm else nn.Identity()

        self.activation = nn.GELU() if use_activation else nn.Identity()
        
    def forward(
        self, 
        features: torch.Tensor, 
        saliency_map: torch.Tensor
    ) -> torch.Tensor:

        attended_features = self.asram(features, saliency_map)

        if isinstance(self.norm, nn.LayerNorm):
            B, C, H, W = attended_features.shape
            attended_features = attended_features.permute(0, 2, 3, 1) 
            attended_features = self.norm(attended_features)
            attended_features = attended_features.permute(0, 3, 1, 2) 
        else:
            attended_features = self.norm(attended_features)

        attended_features = self.activation(attended_features)
        
        return attended_features


def create_asram_attention(
    channels: int,
    reduction_ratio: int = 16,
    kernel_sizes: Tuple[int, ...] = (1, 3, 5),
    dilation_rates: Tuple[int, ...] = (4, 8, 16)
) -> ASRAMAttention:
    return ASRAMAttention(
        channels=channels,
        reduction_ratio=reduction_ratio,
        kernel_sizes=kernel_sizes,
        dilation_rates=dilation_rates
    )


def create_asram_bridge(
    channels: int,
    reduction_ratio: int = 16,
    use_norm: bool = True,
    use_activation: bool = False
) -> ASRAMBridge:
    return ASRAMBridge(
        channels=channels,
        reduction_ratio=reduction_ratio,
        use_norm=use_norm,
        use_activation=use_activation
    )