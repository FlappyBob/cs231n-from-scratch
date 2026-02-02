import copy
from einops import rearrange
from torch import einsum

from torch import nn
import torch
import torch.nn.functional as F
import math

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        Args:
            t: (batch_size,) 时间步，整数或浮点数
        Returns:
            emb: (batch_size, dim) 位置编码
        """
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]  # (batch_size,
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # (batch_size, dim)
        return emb
    
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.dim = dim
        self.scale = nn.Parameter(torch.ones(1, dim, 1, 1)) * (dim ** 0.5) 
        
    def forward(self, x):
        norm_x = F.normalize(x, dim=1)
        return norm_x * self.scale        
    
class Block(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        
        self.norm = RMSNorm(dim_in)
        self.proj = nn.Conv2d(dim_in, dim_out, 3, padding=1)
        self.act = nn.GELU()
        
    def forward(self, x , scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        
        if (exists(scale_shift)):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
    
        x = self.act(x)
        return x
    
class ResnetBlock(nn.Module):
    def __init__(self, dim_in, dim_out, context_dim):
        super().__init__()
        self.dim = dim_in 
        self.dim_out = dim_out
        self.context_dim = context_dim

        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(context_dim, dim_out * 2))
            if exists(context_dim)
            else None
        )
        
        self.block1 = Block(dim_in, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = (
            nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()
        )
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, context=None):
        scale_shift = None
        if (exists(self.mlp)) and (exists(context)):
            context = self.mlp(context)
            context = rearrange(context, 'b c -> b c 1 1')
            scale_shift = context.chunk(2, dim=1)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.dropout(h)
        h = self.block2(h)
        return h + self.res_conv(x)
    
class Unet(nn.Module):
    def Downsample(dim_in, dim_out):
        return nn.Conv2d(dim_in, default(dim_out, dim_in), kernel_size=4, stride=2, padding=1)
    
    def Upsample(dim_in, dim_out):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(dim_in, default(dim_out, dim_in), 3, padding=1)
        )
        
    def __init__(self,
                 dim,
                 condition_dim,
                 dim_mults=(1, 2, 4, 8),
                 channels=3,
                 uncond_prob=0.2):
        super().__init__()
        self.channels = channels
        self.init_conv = nn.Conv2d(channels, dim, 3, padding=1)
        
        dims = [dim] + [dim * m for m in dim_mults]
        in_out = list(zip(dims[:-1], dims[1:]))
        in_out_rev = [(b, a) for (a, b) in reversed(in_out)]
        
        context_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim),
        )
        
        self.condition_dim = condition_dim
        self.condition_mlp = nn.Sequential(
            nn.Linear(condition_dim, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim),
        )
        
        self.uncond_prob = uncond_prob
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        
        # Downsampling blocks
        ####################################################################
        for ind, (dim_in, dim_out) in enumerate(in_out):
            down_block = None
            ##################################################################
            # TODO: Create one UNet downsampling layer `down_block` as a ModuleList.
            # It should be a ModuleList of 3 blocks [ResnetBlock, ResnetBlock, Downsample].
            # Each ResnetBlock operates on dim_in channels and outputs dim_in channels.
            # Make sure to pass the context_dim to each ResnetBlock.
            # The Downsample block operates on dim_in channels and outputs dim_out channels.
            # Make sure to exactly follow this structure of ModuleList in order to
            # load a pretrained checkpoint.
            ##################################################################
            down_block = nn.ModuleList([
                ResnetBlock(dim_in, dim_in, context_dim),
                ResnetBlock(dim_in, dim_in, context_dim),
                Unet.Downsample(dim_in, dim_out)
            ])
            ##################################################################
            self.downs.append(down_block)
            
        # middle blocks
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, context_dim)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, context_dim)
        
        
        # upsampling blocks
        for ind, (dim_in, dim_out) in enumerate(in_out_rev):
            up_block = nn.ModuleList([
                Unet.Upsample(dim_in, dim_out),
                ResnetBlock(dim_out * 2, dim_out, context_dim),
                ResnetBlock(dim_out * 2, dim_out, context_dim),
            ])
            self.ups.append(up_block)
        
        self.final_conv = nn.Conv2d(dim, channels, 1)
        
    def cfg_forward(self, x, time, model_kwargs={}):
        cfg_scale = model_kwargs.pop("cfg_scale")
        print("Classifier-free guidance scale:", cfg_scale)
        model_kwargs = copy.deepcopy(model_kwargs)
        
        
        return x

    def forward(self, x, time, model_kwargs={}):
        """Forward pass through the U-Net.
        Args:
            x: Input tensor of shape (batch_size, channels, height, width).
            time: Tensor of time steps of shape (batch_size,).
            model_kwargs: A dictionary of additional model inputs including
                "text_emb" (text embedding) of shape (batch_size, condition_dim).

        Returns:
            x: Output tensor of shape (batch_size, channels, height, width).
        """

        if "cfg_scale" in model_kwargs:
            return self.cfg_forward(x, time, model_kwargs=model_kwargs)
        
        context = self.time_mlp(time)
        cond_emb = model_kwargs.get("text_emb", None)
        
        if not exists(cond_emb):
            cond_emb = torch.zeros(x.shape[0], self.condition_dim, device=x.device)
            
        if self.training: 
            mask = (torch.rand(cond_emb.shape[0]) > self.uncond_prob).float()
            mask = mask[:, None].to(cond_emb.device)  # B x 1
            cond_emb = cond_emb * mask 
        context = context + self.condition_mlp(cond_emb)
        
        x = self.init_conv(x)
        
        skips = []
        for r1, r2, downsample in self.downs:
            x = r1(x, context=context)
            skips.append(x)
            x = r2(x, context=context)
            skips.append(x)
            x = downsample(x)
            
        x = self.mid_block1(x, context)
        x = self.mid_block2(x, context)

        for upsample, r1, r2 in self.ups:
            x = upsample(x)
            skip1 = skips.pop()
            x = torch.cat((x, skip1), dim=1)
            x = r1(x, context=context)
            
            skip2 = skips.pop()
            x = torch.cat((x, skip2), dim=1)
            x = r2(x, context=context)
            
        x = self.final_conv(x)
        return x
        