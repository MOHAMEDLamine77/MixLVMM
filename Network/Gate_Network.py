import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict, Any, List
from enum import Enum
import numpy as np


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
        d_state: int = 8, 
        d_conv: int = 3,
        expand: int = 1.5,  
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


class FVMEncoderBlock(nn.Module):
    
    def __init__(
        self,
        dim: int,
        dim_out: int,
        vss_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        self.dim = dim
        self.dim_out = dim_out
        self.vss_config = {
            'd_state': 8,
            'd_conv': 3,
            'expand': 1.5,
            'dropout': 0.1,
            'conv_bias': True,
            'bias': False
        }
        if vss_config:
            self.vss_config.update(vss_config)

        self.patch_merge = PatchMerging(dim, dim_out)
        self.vss_block = VSSBlock(
            dim=dim_out,
            **self.vss_config
        )

        self.norm = nn.LayerNorm(dim_out)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x_merged = self.patch_merge(x)  

        x_vss = self.vss_block(x_merged)

        x_out = self.norm(x_vss)
        
        return x_out


class GateNetworkEncoder(nn.Module):
    
    def __init__(
        self,
        input_channels: int = 3,
        base_channels: int = 16,  
        depths: List[int] = [1, 1, 1, 1, 1, 1] 
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.base_channels = base_channels
        self.depths = depths
        self.dims = [base_channels * (2 ** i) for i in range(6)]

        self.initial_conv = nn.Sequential(
            DepthwiseSeparableConv2d(input_channels, base_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        self.fvm_blocks = nn.ModuleList()
        current_dim = base_channels
        
        for i, depth in enumerate(depths):
            target_dim = self.dims[i]

            for j in range(depth):
                if j == 0:
                    block = FVMEncoderBlock(current_dim, target_dim)
                    current_dim = target_dim
                else:
                    block = FVMEncoderBlock(target_dim, target_dim)
                
                self.fvm_blocks.append(block)

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.feature_dim = self.dims[-1]  

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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.initial_conv(x)  
        B, C, H, W = x.shape

        x = x.permute(0, 2, 3, 1) 

        for fvm_block in self.fvm_blocks:
            x = fvm_block(x) 

        x = x.permute(0, 3, 1, 2) 
        x = self.global_pool(x)  
        x = x.flatten(1) 
        
        return x


class SiameseGateNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        feature_dim: int = 128,
        num_experts: int = 2
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.feature_dim = feature_dim
        self.num_experts = num_experts
        self.encoder = GateNetworkEncoder(
            input_channels=input_channels,
            base_channels=16 
        )

        self.feature_projection = nn.Sequential(
            DepthwiseSeparableLinear(self.encoder.feature_dim, feature_dim),
            nn.LayerNorm(feature_dim), 
            nn.ReLU(inplace=True),
            DepthwiseSeparableLinear(feature_dim, feature_dim)
        )

        self.l2_norm = nn.functional.normalize
        self.temperature = nn.Parameter(torch.tensor(1.0))

        self.expert_anchors = nn.Parameter(torch.randn(num_experts, feature_dim))
        nn.init.xavier_uniform_(self.expert_anchors)

        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward_one(self, x: torch.Tensor) -> torch.Tensor:

        features = self.encoder(x)  
        features = self.feature_projection(features) 

        features = self.l2_norm(features, p=2, dim=1) 
        
        return features
    
    def forward(
        self, 
        anchor: torch.Tensor, 
        positive: torch.Tensor, 
        negative: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        anchor_features = self.forward_one(anchor)
        positive_features = self.forward_one(positive)
        negative_features = self.forward_one(negative)
        
        return anchor_features, positive_features, negative_features
    
    def compute_similarity(
        self, 
        features1: torch.Tensor, 
        features2: torch.Tensor
    ) -> torch.Tensor:

        similarity = torch.sum(features1 * features2, dim=1)
        return similarity
    
    def route_to_expert(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        features = self.forward_one(x) 

        normalized_anchors = self.l2_norm(self.expert_anchors, p=2, dim=1)  
        distances = torch.cdist(features, normalized_anchors, p=2)  

        similarities = -distances
        expert_weights = F.softmax(similarities / self.temperature, dim=1)  

        expert_indices = torch.argmax(expert_weights, dim=1) 
        
        return expert_weights, expert_indices
    
    def get_expert_assignment(self, x: torch.Tensor) -> int:
        with torch.no_grad():
            _, expert_indices = self.route_to_expert(x)
            return expert_indices.item()


class TripletLoss(nn.Module):
    
    def __init__(self, margin: float = 1.0, distance_function: str = 'cosine'):
        super().__init__()
        self.margin = margin
        self.distance_function = distance_function
        
    def forward(
        self, 
        anchor: torch.Tensor, 
        positive: torch.Tensor, 
        negative: torch.Tensor
    ) -> torch.Tensor:
        if self.distance_function == 'cosine':
            pos_distance = 1 - F.cosine_similarity(anchor, positive, dim=1)
            neg_distance = 1 - F.cosine_similarity(anchor, negative, dim=1)
        else:  
            pos_distance = F.pairwise_distance(anchor, positive, p=2)
            neg_distance = F.pairwise_distance(anchor, negative, p=2)

        loss = F.relu(pos_distance - neg_distance + self.margin)
        
        return loss.mean()


class ContrastiveLoss(nn.Module):

    def __init__(self, margin: float = 2.0):
        super().__init__()
        self.margin = margin
        
    def forward(
        self, 
        features1: torch.Tensor, 
        features2: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:

        distances = F.pairwise_distance(features1, features2, p=2)
        
        loss = (1 - labels) * torch.pow(distances, 2) + \
               labels * torch.pow(F.relu(self.margin - distances), 2)
        
        return loss.mean()


def create_siamese_gate_network(
    input_channels: int = 3,
    feature_dim: int = 512, 
    num_experts: int = 2
) -> SiameseGateNetwork:
    return SiameseGateNetwork(
        input_channels=input_channels,
        feature_dim=feature_dim,
        num_experts=num_experts
    )


def create_triplet_loss(margin: float = 1.0) -> TripletLoss:
    return TripletLoss(margin=margin, distance_function='cosine')