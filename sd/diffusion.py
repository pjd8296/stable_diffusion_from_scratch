import torch
import torch.nn as nn
import torch.nn.functional as F
from sd.attention import SelfAttention, CrossAttention


class TimeEmbedding(nn.Module):
    
    def __init__(self, n_embed: int):
        super().__init__()
        
        self.linear_1 = nn.Linear(n_embed, 4 * n_embed)
        self.linear_2 = nn.Linear(4 * n_embed, n_embed)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.linear_1(x)    # x: (1, 320)
        x = F.silu(x)
        x = self.linear_2(x)    # x: (1, 1280)
        
        return x


class Unet_Residual_Block(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int, n_time=1280):
        super().__init__()
        
        self.gn_feat = nn.GroupNorm(32, in_channels)
        self.conv_feat = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)
        
        self.gn_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        
    def forward(self, feature, time):
        # feature: (batch, in_channels, H, W)
        # time: (1, 1280)
        residue = feature
        
        feature = self.gn_feat(feature)
        feature = F.silu(feature)
        feature = self.conv_feat(feature)
        
        time = F.silu(time)
        time = self.linear_time(time)
        
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        
        merged = self.gn_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)
        
        return merged + self.residual(residue)


class Unet_Attention_Block(nn.Module):
    def __init__(self, n_head: int, n_embed: int, d_context=768):
        super().__init__()
        
        channels = n_head * n_embed
        self.gn = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        
        self.ln_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.ln_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.ln_3 = nn.LayerNorm(channels)
        self.linear_1 = nn.Linear(channels, 4 * channels)
        self.linear_2 = nn.Linear(4 * channels, channels)
        
        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        
    def forward(self, x, context):
        
        # x: (batch, features, H, W)
        # context: (batch, seq_len, dim)
        residue_long = x
        
        x = self.gn(x)
        x = self.conv_input(x)
        
        n, c, h, w = x.shape
        x = x.view((n, c, h*w))     # (batch, features, H, W) -> (batch, features, H * W)
        x = x.transpose(-1, -2)     # (batch, H * W, features)
        
        # Normalization + self-attention with skip connection
         
        residue_short = x
        
        x = self.ln_1(x)
        self.attention_1(x)
        x += residue_short
        
        residue_short = x
        
        # Normalization + cross-attention with skip connection
        x = self.ln_2(x)
        self.attention_2(x, context)
        x += residue_short
        
        residue_short = x
        
        # Normalization + FF with GeLU with skip connection
        x = self.ln_3(x)
        
        x, gate = self.linear_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        
        x = self.linear_2(x)
        x += residue_short
        
        x = x.transpose(-1, -2)     # (batch, H * W, features) -> (batch, features, H * W)
        x = x.view((n, c, h, w))
        
        return self.conv_output(x) + residue_long
        
        return x


class Upsample(nn.Module):
    
    def __init__(self, channels: int):
        
        super().__init__()
        
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        
        # (batch, features, H, W) -> (batch, features, H * 2, W * 2)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        
        return self.conv(x)


class SwitchSequential(nn.Sequential):
    
    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        
        for layer in self:
            if isinstance(layer, Unet_Attention_Block):
                x = layer(x, context)
            elif isinstance(layer, Unet_Residual_Block):
                x = layer(x, time)
            else:
                x = layer(x)
        
        return x
    

class Unet(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.encoders = nn.ModuleList([
            # (batch, 4, H/8, W/8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            
            SwitchSequential(Unet_Residual_Block(320, 320), Unet_Attention_Block(8, 40)),
            SwitchSequential(Unet_Residual_Block(320, 320), Unet_Attention_Block(8, 40)),
            
            # (batch, 320, H/8, W/8) ->  (batch, 320, H/16, W/16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            
            SwitchSequential(Unet_Residual_Block(320, 640), Unet_Attention_Block(8, 80)),
            SwitchSequential(Unet_Residual_Block(640, 640), Unet_Attention_Block(8, 80)),
            
            # (batch, 640, H/16, W/16) ->  (batch, 640, H/32, W/32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            
            SwitchSequential(Unet_Residual_Block(640, 1280), Unet_Attention_Block(8, 160)),
            SwitchSequential(Unet_Residual_Block(1280, 1280), Unet_Attention_Block(8, 160)),
            
            # (batch, 1280, H/32, W/32) ->  (batch, 1280, H/64, W/64)
            SwitchSequential(nn.Conv2d(1280, 128, kernel_size=3, stride=2, padding=1)),
            
            # (batch, 1280, H/64, W/64) -> (batch, 1280, H/64, W/64)
            SwitchSequential(Unet_Residual_Block(1280, 1280)),
            SwitchSequential(Unet_Residual_Block(1280, 1280))
        ])
        
        self.bottleneck = SwitchSequential(
            Unet_Residual_Block(1280, 1280),
            Unet_Attention_Block(8, 160),
            Unet_Residual_Block(1280, 1280)
        )
        
        self.decoders = nn.ModuleList([
            # (batch, 2560, H/64, W/64) ->  (batch, 1280, H/64, W/64)
            SwitchSequential(Unet_Residual_Block(2560, 1280)),
            SwitchSequential(Unet_Residual_Block(2560, 1280)),
            SwitchSequential(Unet_Residual_Block(2560, 1280), Upsample(1280)),
            
            SwitchSequential(Unet_Residual_Block(2560, 1280), Unet_Attention_Block(8, 160)),
            SwitchSequential(Unet_Residual_Block(2560, 1280), Unet_Attention_Block(8, 160)),
            
            SwitchSequential(Unet_Residual_Block(1920, 1280), Unet_Attention_Block(8, 160), Upsample(1280)),
            SwitchSequential(Unet_Residual_Block(1920, 640), Unet_Attention_Block(8, 80)),
            SwitchSequential(Unet_Residual_Block(1280, 640), Unet_Attention_Block(8, 80)),
            
            SwitchSequential(Unet_Residual_Block(960, 640), Unet_Attention_Block(8, 80), Upsample(640)),
            SwitchSequential(Unet_Residual_Block(960, 320), Unet_Attention_Block(8, 40)),
            SwitchSequential(Unet_Residual_Block(640, 320), Unet_Attention_Block(8, 80)),
            
            SwitchSequential(Unet_Residual_Block(640, 320), Unet_Attention_Block(8, 40))
        ])
        
class Unet_output_layer(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.gn = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        # x: (batch, 320, H/8, W/8)
        x = self.gn(x)
        x = F.silu(x)
        x = self.conv(x)
        
        return x    # (batch, 4, H/8, W/8)


class Diffusion(nn.Module):
    
    def __init__(self):
        
        self.time_embed = TimeEmbedding(320)
        self.unet = Unet()
        self.final = Unet_output_layer(320, 4)
    
    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor):

        # latent : (batch, 4, H/8, W/8)
        # context : (batch, seq_len, dim)
        # time : (1, 320)
        
        time_embedding = self.time_embed(time)      # (1, 320) -> (1, 1280)
        output = self.unet(latent, context, time_embedding)   # (batch, 4, H/8, W/8) -> (batch, 320, H/8, W/8)
        output = self.final(output)                 # (batch, 320, H/8, W/8) -> (batch, 4, H/8, W/8)
        
        return output