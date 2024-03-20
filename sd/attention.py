import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfAttention(nn.Module):
    
    def __init__(self, num_heads: int, embed_dim: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        
        self.in_proj = nn.Linear(embed_dim, 3*embed_dim, bias=in_proj_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=out_proj_bias)
        
        self.num_heads = num_heads
        self.dim_head = embed_dim // num_heads
        
    def forward(self, x: torch.Tensor, causal_mask=False):
        
        # x: (batch, seq_lem, dim)
        
        input_shape = x.shape
        batch, seq_len, embed_dim = input_shape
        
        interim_shape = (batch, seq_len, self.num_heads, self.dim_head)
        
        # (batch, seq_len, dim) -> (batch, seq_len, dim*3)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        
        # (batch, seq_len, dim) -> (batch, seq_len, num_heads, dim / num_heads) -> (batch, num_heads, seq_len, dim / num_heads)
        q = q.view(interim_shape).transpose(1,2)
        k = k.view(interim_shape).transpose(1,2)
        v = v.view(interim_shape).transpose(1,2)
        
        # (batch, num_heads, seq_len, seq_len)
        weight = q @ k.transpose(-1,-2)
        
        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill(mask, -torch.inf) # later softmax will fill the masked region with zeros
        
        weight /= math.sqrt(self.num_heads)
        weight = F.softmax(weight, dim=-1)
        
        # (batch, num_heads, seq_len, seq_len) @ (batch, num_heads, seq_len, dim / num_heads) -> (batch, num_heads, seq_len, dim / num_heads)
        output = weight @ v
        
        output = output.transpose(1, 2)
        
        output = output.reshape(input_shape)
        
        output = self.out_proj(output)
        
        # (batch, seq_len, dim)
        return output
        

class CrossAttention(nn.Module):
    
    def __init__(self, num_heads: int, embed_dim: int, d_cross: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, embed_dim, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, embed_dim, bias=in_proj_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=out_proj_bias)
        
        self.num_heads = num_heads
        self.dim_head = embed_dim // num_heads
        
    def forward(self, x: torch.Tensor, y):
        
        # x: latent: (batch, seq_len_q, dim_q)
        # x: context: (batch, seq_len_kv, dim_kv) = (batch, 77, 768)
        
        
        input_shape = x.shape
        batch, seq_len, embed_dim = input_shape
        
        interim_shape = (batch, -1, self.num_heads, self.dim_head)
        
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)
        
        # (batch, seq_len, dim) -> (batch, seq_len, num_heads, dim / num_heads) -> (batch, num_heads, seq_len, dim / num_heads)
        q = q.view(interim_shape).transpose(1,2)
        k = k.view(interim_shape).transpose(1,2)
        v = v.view(interim_shape).transpose(1,2)
        
        # (batch, num_heads, seq_len, seq_len)
        weight = q @ k.transpose(-1,-2)
        
        weight /= math.sqrt(self.num_heads)
        weight = F.softmax(weight, dim=-1)
        
        # (batch, num_heads, seq_len, seq_len) @ (batch, num_heads, seq_len, dim / num_heads) -> (batch, num_heads, seq_len, dim / num_heads)
        output = weight @ v
        
        output = output.transpose(1, 2).contiguous()
        
        output = output.view(input_shape)
        
        output = self.out_proj(output)
        
        # (batch, seq_len, dim)
        return output