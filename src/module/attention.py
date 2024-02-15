"""
Low precision attention modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.module.base import _QBase, _QBaseLinear
from src.module.fuse import MulShift

class QAttention(nn.Module):
    def __init__(
            self,
            dim:int,
            num_heads, 
            qkv_bias=False,
            qk_norm=False, 
            attn_drop=0.0,
            proj_drop=0.0,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        assert dim % num_heads == 0,"dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # attention scale
        self.scale = self.head_dim ** (-0.5)

        self.qkv = _QBaseLinear(dim, int(dim*3), bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        # dropout
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = _QBaseLinear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_scale = MulShift()
        self.attn_scale.scale.data.copy_(self.scale)

        # quantizers 
        self.xq = _QBase(nbit=32)
        self.qqkv = _QBase(nbit=32)
    
    def forward(self, x:torch.Tensor):
        B, N, C = x.shape

        x = self.xq(x)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # reshape to (qkv), batch, num_heads, token, head_dim
        qkv = self.qqkv(qkv)

        q, k, v = qkv.unbind(0) # batch, num_heads, token, head_dim
        q, k = self.q_norm(q), self.k_norm(k)

        # attention
        attn = q @ k.transpose(-2, -1)  # out dim = token x token
        attn = self.attn_scale(attn)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v # out dim = token x head_dim

        # reshape
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    