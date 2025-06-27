import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union, Dict, Any
from abc import ABC, abstractmethod
from enum import Enum
import math
import warnings


class DepthwiseSeparableLinear(nn.Module):
    
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()

        self.pointwise = nn.Linear(in_features, out_features, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(x)


class ScalingOperation(Enum):
    MERGE = "merge"     
    EXPAND = "expand"   


class PatchOperationBase(nn.Module, ABC):
    
    def __init__(self, dim: int, dim_out: int):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
    
    @property
    @abstractmethod
    def scaling_factor(self) -> int:
        pass


class PatchMerging(PatchOperationBase):
    
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        norm_layer: nn.Module = nn.LayerNorm
    ):
        dim_out = dim_out or (2 * dim)
        super().__init__(dim, dim_out)
        
        self.norm = norm_layer(4 * dim)          

        self.reduction = DepthwiseSeparableLinear(4 * dim, dim_out, bias=False)

        self._initialize_weights()
        
    def _initialize_weights(self) -> None:
        if hasattr(self.reduction.pointwise, 'weight'):
            nn.init.xavier_uniform_(self.reduction.pointwise.weight)
        
    @property
    def scaling_factor(self) -> int:
        return 2
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        if H % 2 != 0 or W % 2 != 0:
            warnings.warn(
                f"Input dimensions ({H}, {W}) not divisible by 2. "
                "Padding will be applied."
            )
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


class PatchExpanding(PatchOperationBase):
    
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        norm_layer: nn.Module = nn.LayerNorm
    ):
        dim_out = dim_out or (dim // 2)
        super().__init__(dim, dim_out)
        
        self.expand_dim = 4 * dim_out  
        self.norm = norm_layer(dim)

        self.expansion = DepthwiseSeparableLinear(dim, self.expand_dim, bias=False)

        self._initialize_weights()
        
    def _initialize_weights(self) -> None:
        if hasattr(self.expansion.pointwise, 'weight'):
            nn.init.xavier_uniform_(self.expansion.pointwise.weight)
            with torch.no_grad():
                self.expansion.pointwise.weight.mul_(0.1)
            
    @property
    def scaling_factor(self) -> int:
        return 2
        
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
            dim_out = (2 * dim) if operation == ScalingOperation.MERGE else (dim // 2)
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

        try:
            from .VSS_Block import DualVSSBlock
            from .ASRAM import ASRAMAttention
        except ImportError:
            try:
                from VSS_Block import DualVSSBlock
                from ASRAM import ASRAMAttention
            except ImportError:
                print("Warning: VSS and ASRAM modules not found. Creating mock implementations for testing.")
                DualVSSBlock = self._create_mock_dual_vss()
                ASRAMAttention = self._create_mock_asram()

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
        
    def _create_mock_dual_vss(self):
        class MockDualVSSBlock(nn.Module):
            def __init__(self, dim, **kwargs):
                super().__init__()
                self.norm = nn.LayerNorm(dim)
                
            def forward(self, x):
                return x + self.norm(x)
        return MockDualVSSBlock
        
    def _create_mock_asram(self):
        class MockASRAMAttention(nn.Module):
            def __init__(self, channels, **kwargs):
                super().__init__()
                self.norm = nn.LayerNorm(channels)
                
            def forward(self, features, saliency_map):
                B, C, H, W = features.shape
                features = features.permute(0, 2, 3, 1)  
                features = self.norm(features)
                return features.permute(0, 3, 1, 2)  
        return MockASRAMAttention
        
    def _apply_gradient_checkpointing(
        self, 
        func: callable, 
        *args, **kwargs
    ) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(func, *args, **kwargs)
        return func(*args, **kwargs)
        
    def forward(
        self, 
        x: torch.Tensor, 
        saliency_map: torch.Tensor
    ) -> torch.Tensor:

        x_scaled = self.patch_op(x)  
        B, H_new, W_new, C_new = x_scaled.shape

        saliency_adapted = self._adapt_saliency_map(
            saliency_map, (H_new, W_new)
        )

        x_vss = x_scaled

        x_vss = self._apply_gradient_checkpointing(
            self.vss_blocks, x_vss
        )

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


class FVMBlockFactory:
    
    @staticmethod
    def create_encoder_block(
        dim: int,
        level: int = 0,
        use_checkpoint: bool = False
    ) -> FocusVisionMambaBlock:

        vss_config = {
            'd_state': 16 + (level * 4),  
            'dropout': 0.1 if level > 2 else 0.0, 
        }
        
        asram_config = {
            'reduction_ratio': max(8, 16 - level * 2),  
        }
        
        return FocusVisionMambaBlock(
            dim=dim,
            operation=ScalingOperation.MERGE,
            vss_config=vss_config,
            asram_config=asram_config,
            use_checkpoint=use_checkpoint
        )
        
    @staticmethod
    def create_decoder_block(
        dim: int,
        level: int = 0,
        use_checkpoint: bool = False
    ) -> FocusVisionMambaBlock:
        vss_config = {
            'd_state': 20,  
            'expand': 3,    
        }
        
        asram_config = {
            'reduction_ratio': 8,  
            'kernel_sizes': (1, 3, 5, 7), 
        }
        
        return FocusVisionMambaBlock(
            dim=dim,
            operation=ScalingOperation.EXPAND,
            vss_config=vss_config,
            asram_config=asram_config,
            use_checkpoint=use_checkpoint
        )


def create_fvm_encoder_block(
    dim: int, 
    level: int = 0, 
    use_checkpoint: bool = False
) -> FocusVisionMambaBlock:
    return FVMBlockFactory.create_encoder_block(dim, level, use_checkpoint)


def create_fvm_decoder_block(
    dim: int, 
    level: int = 0, 
    use_checkpoint: bool = False
) -> FocusVisionMambaBlock:
    return FVMBlockFactory.create_decoder_block(dim, level, use_checkpoint)