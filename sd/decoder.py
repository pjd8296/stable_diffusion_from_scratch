import torch
import torch.nn as nn
import torch.nn.functional as F
from sd.attention import SelfAttention


class VAEAttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super(). __init__()
        
        self.gn = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #  x: (batch, features, H, W)
        
        residue = x
        n, c, h, w = x.shape
        
        x = x.view(n, c, h*w)
        # x: (batch, features, H*W) -> x: (batch, H*W, features)
        x = x.transpose(-1, -2)
        
        x = self.attention(x)
        
        # x: (batch, H*W, features) -> x: (batch, features, H*W)
        x = x.transpose(-1, -2)
        
        #  x: (batch, features, H, W)
        x = x.view(n, c, h, w)
        
        x += residue
        
        return x
           

class VAEResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.gn_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.gn_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, H, W)
        
        residue = x
        
        x = self.gn_1(x)
        x = F.silu(x)
        
        x = self.conv_1(x)
        x = self.gn_2(x)
        x = F.silu(x)
        
        x = self.conv_2(x)
        
        return x + self.residual_layer(residue)

    

class VAEDecoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            VAEResidualBlock(512, 512),
            VAEAttentionBlock(512),
            
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),     # (batch, 512, H/8, W/8) -> (batch, 512, H/8, W/8)
            
            nn.Upsample(scale_factor=2),    # (batch, 512, H/8, W/8) -> (batch, 512, H/4, W/4)
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            
            nn.Upsample(scale_factor=2),    # (batch, 512, H/4, W/4) -> (batch, 512, H/2, W/2)
            
            VAEResidualBlock(512, 256),
            VAEResidualBlock(256, 256),
            VAEResidualBlock(256, 256),
            
            nn.Upsample(scale_factor=2),    # (batch, 256, H/2, W/2) -> (batch, 256, H, W)
            
            VAEResidualBlock(256, 128),
            VAEResidualBlock(128, 128),
            VAEResidualBlock(256, 128),
            
            nn.GroupNorm(32, 128),
            
            nn.SiLU(),
            
            nn.Conv2d(128, 3, kernel_size=3, padding=1)     # (batch, 128, H, W) -> (batch, 3, H, W)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # x: (batch, 4, H/8, W/8)
        x /= 0.18215        # reverse the scaling used during encoding
        
        for module in self:
            x = module(x)
            
        return x    # (batch, 3, H, W)
        