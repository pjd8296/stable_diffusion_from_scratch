import torch
import torch.nn as nn
import torch.nn.functional as F
from decoder import VAEAttentionBlock, VAEResidualBlock


class VAEEncoder(nn.Sequential):
    
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            VAEResidualBlock(128, 128),
            VAEResidualBlock(128, 128),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            
            VAEResidualBlock(128, 256),
            VAEResidualBlock(256, 256),
            
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            
            VAEResidualBlock(256, 512),
            VAEResidualBlock(512, 512),
            
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            
            VAEAttentionBlock(512),
            
            VAEResidualBlock(512, 512),
            
            nn.GroupNorm(32, 512),
            
            nn.SiLU(),
            
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            
            nn.Conv2d(8, 8, kernel_size=1, padding=0)
        )
        
    
    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        
        # x: (batch, in_channel, height, width)
        # noise: (batch, out_channel, height/8, width/8)
        
        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                # (padding_left, padding_right, padding_top, padding_bottom)
                x = F.pad(x, (0,1,0,1))
            x = module(x)
            
        # (batch, 8, H, H/8, W/8) -> Two tensors of shape (B, 4, H/8, W/8)
        mean, log_var = torch.chunk(x, 2, dim=1)
        
        log_var = torch.clamp(log_var, -30, 20)
        
        # transform log variance into variance
        var = log_var.exp()
        
        std = var.sqrt()
        
        # convert N(0, 1) to N(mean, var) : X = mean + std * Z, where Z -> noise
        x = mean + std * noise
        
        # scale the output by a constant given in the paper
        x *= 0.18215
        
        return x