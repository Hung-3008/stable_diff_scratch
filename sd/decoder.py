import torch 
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention



class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(channels)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch size, Channels, Height, Width)
        residue = x 
        n, c, h, w = x.shape
        # (Batch size, Channels, Height, Width) -> (Batch size, Channels, Height*Width)
        x = x.view(n, c, h*w)
        # (Batch size, Channels, Height*Width) -> (Batch size, Height*Width, Channels)
        x = x.transpose(-1,-2)
        # (Batch size, Height*Width, Channels) -> (Batch size, Height*Width, Channels)
        x = self.attention(x)
        # (Batch size, Height*Width, Channels) -> (Batch size, Channels, Height*Width)
        x = x.transpose(-1,-2)
        # (Batch size, Channels, Height*Width) -> (Batch size, Channels, Height, Width)
        x = x.view(n, c, h, w)
        x += residue

        return x 






class VAE_ResidualBlock(nn.Module):
    def __init__(self, inchannels, outchannels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, inchannels)
        self.conv_1 = nn.conv2d(inchannels, outchannels, kernel_size=3, padding=1)

        self.groupnorm = nn.GroupNorm(32, outchannels)
        self.conv_2 = nn.conv2d(outchannels, outchannels, kernel_size=3, padding=1)

        if inchannels == outchannels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(inchannels, outchannels, kernel_size=1, padding=0)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch size, In Channels, Height, Width)
        residue = x 

        x = self.groupnorm(x)
        x = F.silu(x)
        x = self.conv_1(x)
        x = self.groupnorm(x) 
        x = F.silu(x)
        x = self.conv_2(x)
        return x + self.residual_layer(residue)