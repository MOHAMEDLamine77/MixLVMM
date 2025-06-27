import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DepthwiseSeparableLinear(nn.Module):    
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.pointwise = nn.Linear(in_features, out_features, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(x)


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

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        dropout: float = 0.0,
        conv_bias: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        self.dim = dim

        self.norm = nn.LayerNorm(dim)

        self.ss2d = SS2D(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            conv_bias=conv_bias,
            bias=bias,
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x_norm = self.norm(x)
        x_ss2d = self.ss2d(x_norm)
        out = x + x_ss2d
        
        return out


class DualVSSBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        dropout: float = 0.0,
        conv_bias: bool = True,
        bias: bool = False,
    ):
        super().__init__()

        self.vss1 = VSSBlock(
            dim=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            conv_bias=conv_bias,
            bias=bias,
        )
        
        self.vss2 = VSSBlock(
            dim=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            conv_bias=conv_bias,
            bias=bias,
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vss1(x)
        x = self.vss2(x)
        return x