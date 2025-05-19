import torch
from torch import nn
from .block import GroupNorm32, TimeEmbedding_new, AttentionBlock, Upsample, Downsample
import math

class ConditionInjection(nn.Module):
    def __init__(self, n_channels, size, input_size = 128, data_channel = 2):
        """
        Injects conditional information into the network.
        """
        super().__init__()
        
        # cond_vector [batch_size, n_channels, 128, 128] 
        # --> [batch_size, n_channels, size, size]
        self.data_channel = data_channel
        downsample_steps = int(math.log2(input_size // size))  # 计算从128降到size所需要的下采样步数
        self.norm = GroupNorm32(n_channels)
        # self.cond_projection = nn.Sequential(
        #     nn.Conv2d(data_channel, n_channels, kernel_size=3, padding=1),  # 保持分辨率
        #     nn.SiLU(),
            
        #     # 下采样操作
        #     *[nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=2, padding=1) for _ in range(downsample_steps)],  # 逐步下采样
            
        #     nn.SiLU(),
            
        #     # 进一步卷积处理
        #     nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1),
        #     nn.SiLU(),
        # )
        self.cond_projection_1 = nn.Sequential(
            # 使用最大池化进行下采样
            *[nn.MaxPool2d(kernel_size=2, stride=2) for _ in range(downsample_steps)],  # 逐步下采样
            nn.SiLU(),

            # # 进一步卷积处理
            # nn.Conv2d(data_channel, n_channels, kernel_size=3, padding=1),
            # nn.SiLU(),
        )
        self.feature_projection_1 = nn.Linear(n_channels, n_channels * 2)
        self.feature_projection_2 = nn.Linear(data_channel, n_channels)
        
        self.output = nn.Linear(n_channels, n_channels)
        self.scale = 1 / math.sqrt(math.sqrt(n_channels))

    def forward(self, x, cond_matrix):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`.
        * `cond_matrix` has shape `[batch_size, n_channels, 128, 128]`.
        """
        batch_size, n_channels, height, width = x.shape

        # Condition projection
        q_ori = self.cond_projection_1(cond_matrix).view(batch_size, self.data_channel, -1).permute(0, 2, 1) # (B, size*size, n_channels)
        q = self.feature_projection_2(q_ori)

        # Feature projection
        h2 = self.norm(x).view(batch_size, n_channels, -1).permute(0, 2, 1)
        kv = self.feature_projection_1(h2)
        k, v = torch.chunk(kv, 2, dim=-1)

        # Scaled dot-product attention
        attn = torch.einsum('bid,bjd->bij', q * self.scale, k * self.scale)
        attn = attn.softmax(dim=2)
        out = torch.einsum('bij,bjd->bid', attn, v)

        # Reshape and project
        out = out.reshape(batch_size, -1, n_channels)
        out = self.output(out)
        out = out.permute(0, 2, 1).view(batch_size, n_channels, height, width)

        return (out + x) / math.sqrt(2.)
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, dropout=0.1, up=False, down=False):
        """
        * `in_channels` is the number of input channels
        * `out_channels` is the number of output channels
        * `time_channels` is the number channels in the time step ($t$) embeddings
        * `dropout` is the dropout rate
        """
        super().__init__()
        self.norm1 = GroupNorm32(in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = GroupNorm32(out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        # Linear layer for embeddings
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels)
        )

        # BigGAN style: use resblock for up/downsampling
        self.updown = up or down
        if up:
            self.h_upd = Upsample(in_channels, use_conv=False)
            self.x_upd = Upsample(in_channels, use_conv=False)
        elif down:
            self.h_upd = Downsample(in_channels, use_conv=False)
            self.x_upd = Downsample(in_channels, use_conv=False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

    def forward(self, x, t):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        if self.updown:
            h = self.conv1(self.h_upd(self.act1(self.norm1(x))))
            x = self.x_upd(x)
        else:
            h = self.conv1(self.act1(self.norm1(x)))

        # Adaptive Group Normalization
        t_ = self.time_emb(t)[:, :, None, None]
        h = h + t_

        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.shortcut(x)


class ResAttBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, has_attn, attn_channels_per_head, dropout):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels, dropout=dropout)
        if has_attn:
            self.attn = AttentionBlock(out_channels, attn_channels_per_head)
        else:
            self.attn = nn.Identity()

    def forward(self, x, t):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class MiddleBlock(nn.Module):
    def __init__(self, n_channels, time_channels, attn_channels_per_head, dropout):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels, dropout=dropout)   # 全都没有normalization func?
        self.attn = AttentionBlock(n_channels, attn_channels_per_head)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels, dropout=dropout)

    def forward(self, x, t):
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
        return x


class UpsampleRes(nn.Module):
    def __init__(self, n_channels, time_channels, dropout):
        super().__init__()
        self.op = ResidualBlock(n_channels, n_channels, time_channels, dropout=dropout, up=True)

    def forward(self, x, t):
        return self.op(x, t)


class DownsampleRes(nn.Module):
    def __init__(self, n_channels, time_channels, dropout):
        super().__init__()
        self.op = ResidualBlock(n_channels, n_channels, time_channels, dropout=dropout, down=True)

    def forward(self, x, t):
        return self.op(x, t)
 

class UNet_cond(nn.Module):
    def __init__(self, image_shape = [3, 32, 32], n_channels = 128,
                 ch_mults = (1, 2, 2, 2),
                 is_attn = (False, True, False, False),
                 attn_channels_per_head = None,
                 dropout = 0.1,
                 n_blocks = 2,
                 use_res_for_updown = False,
                 ):  
        
        """
        * `image_shape` is the (channel, height, width) size of images.
        * `n_channels` is number of channels in the initial feature map that we transform the image into
        * `ch_mults` is the list of channel numbers at each resolution. The number of channels is `n_channels * ch_mults[i]`
        * `is_attn` is a list of booleans that indicate whether to use attention at each resolution
        * `dropout` is the dropout rate
        * `n_blocks` is the number of `UpDownBlocks` at each resolution
        * `use_res_for_updown` indicates whether to use ResBlocks for up/down sampling (BigGAN-style)
        * `augment_dim` indicates augmentation label dimensionality, 0 = no augmentation
        """
        super().__init__()

        n_resolutions = len(ch_mults)

        self.image_proj = nn.Conv2d(image_shape[0], n_channels, kernel_size=3, padding=1)

        # Embedding layers (time & augment)
        time_channels = n_channels * 4
        self.time_emb = TimeEmbedding_new(time_channels)

        # Down stages
        down = []
        in_channels = n_channels
        h_channels = [n_channels]
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = n_channels * ch_mults[i]
            # `n_blocks` at the same resolution
            down.append(ResAttBlock(in_channels, out_channels, time_channels, is_attn[i], attn_channels_per_head, dropout))
            h_channels.append(out_channels)
            for _ in range(n_blocks - 1):
                down.append(ResAttBlock(out_channels, out_channels, time_channels, is_attn[i], attn_channels_per_head, dropout))
                h_channels.append(out_channels)
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                if use_res_for_updown:
                    down.append(DownsampleRes(out_channels, time_channels, dropout))
                else:
                    down.append(Downsample(out_channels))
                h_channels.append(out_channels)
            in_channels = out_channels
            
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(out_channels, time_channels, attn_channels_per_head, dropout)

        # Up stages
        up = []
        cond, size = [], image_shape[-1]//(2**(n_resolutions-1))
        in_channels = out_channels
        for i in reversed(range(n_resolutions)):
            # Number of output channels at this resolution
            out_channels = n_channels * ch_mults[i]
            # `n_blocks + 1` at the same resolution
            for _ in range(n_blocks + 1):
                cond.append(ConditionInjection(in_channels, size))
                up.append(ResAttBlock(in_channels + h_channels.pop(), out_channels, time_channels, is_attn[i], attn_channels_per_head, dropout))
                in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                if use_res_for_updown:
                    up.append(UpsampleRes(out_channels, time_channels, dropout))
                else:
                    up.append(Upsample(out_channels))
            size = size * 2
        assert not h_channels
        self.up = nn.ModuleList(up)
        self.cond = nn.ModuleList(cond)
        # Final normalization and convolution layer
        self.norm = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        self.final = nn.Conv2d(out_channels, image_shape[0], kernel_size=3, padding=1)

    def forward(self, x, t, cond_matrix):    # 降采样升采样都 embed t，只在升采样embed cond
        # x: [B*(1+horizon), C, H, W].
        # t: [B*(1+horizon)]
        # cond: [B*(1+horizon), C, H, W]
        # scale: [B*(1+horizon)]
        
        t = self.time_emb(t)
    
        
        x = self.image_proj(x)

        # Store outputs for skip connections
        h = [x]

        # Downsample
        for m in self.down:
            if isinstance(m, Downsample):
                x = m(x)
            elif isinstance(m, DownsampleRes):
                x = m(x, t)
            else:
                x = m(x, t).contiguous()
            h.append(x)

        # Middle block
        x = self.middle(x, t).contiguous()

        # Upsample
        count = 0
        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x)
            elif isinstance(m, UpsampleRes):
                x = m(x, t)
            else:
                # Inject conditional information
                x = self.cond[count](x, cond_matrix)
                count += 1
                
                # Concatenate with skip connection
                s = h.pop()
                x = torch.cat([x, s], dim=1)
                x = m(x, t).contiguous()

        # Final layer
        return self.final(self.act(self.norm(x)))


'''
from model.unet import UNet
net = UNet()
import torch
x = torch.zeros(1, 3, 32, 32)
t = torch.zeros(1,)

net(x, t).shape
sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6

>>> 35.746307 M parameters for CIFAR-10 model (original DDPM)
'''