import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2

        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)

        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        return emb

class TimeMLP(nn.Module):
    def __init__(self, time_dim):
        super().__init__()

        self.net = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim)
        )

    def forward(self, t):
        return self.net(t)

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)

        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x * self.weight + self.bias

        return x

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class TimeNAFBlock(nn.Module):
    def __init__(self, channels, time_dim, dw_expand=2, ffn_expand=2):
        super().__init__()

        dw_channels = channels * dw_expand
        ffn_channels = channels * ffn_expand

        self.norm1 = LayerNorm2d(channels)

        self.conv1 = nn.Conv2d(channels, dw_channels, kernel_size=1)
        self.dwconv = nn.Conv2d(
            dw_channels,
            dw_channels,
            kernel_size=3,
            padding=1,
            groups=dw_channels
        )

        self.sg = SimpleGate()

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channels // 2, dw_channels // 2, kernel_size=1)
        )

        self.conv2 = nn.Conv2d(dw_channels // 2, channels, kernel_size=1)

        self.time_proj1 = nn.Linear(time_dim, channels)
        self.norm2 = LayerNorm2d(channels)

        self.conv3 = nn.Conv2d(channels, ffn_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(ffn_channels // 2, channels, kernel_size=1)

        self.time_proj2 = nn.Linear(time_dim, channels)


        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x, t_emb):
        h = self.norm1(x)

        h = self.conv1(h)
        h = self.dwconv(h)
        h = self.sg(h)

        h = h * self.sca(h)
        h = self.conv2(h)

        h = h + self.time_proj1(t_emb)[:, :, None, None]

        x = x + self.beta * h

        h = self.norm2(x)

        h = self.conv3(h)
        h = self.sg(h)
        h = self.conv4(h)

        h = h + self.time_proj2(t_emb)[:, :, None, None]

        x = x + self.gamma * h

        return x

class NAFResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()

        if in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.proj = nn.Identity()

        self.naf = TimeNAFBlock(out_channels, time_dim)

    def forward(self, x, t_emb):
        x = self.proj(x)
        x = self.naf(x, t_emb)
        return x

class DownBlockNAF(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()

        self.block = NAFResBlock(in_channels, out_channels, time_dim)

        self.down = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1
        )

    def forward(self, x, t_emb):
        x = self.block(x, t_emb)
        skip = x
        x = self.down(x)
        return x, skip

class UpBlockNAF(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, time_dim):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.block = NAFResBlock(
            out_channels + skip_channels,
            out_channels,
            time_dim
        )

    def forward(self, x, skip, t_emb):
        x = self.up(x)

        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        x = torch.cat([x, skip], dim=1)
        x = self.block(x, t_emb)

        return x

class ConditionEncoderNAF(nn.Module):
    def __init__(self, condition_channels=6, base_channels=64, time_dim=256):
        super().__init__()

        self.in_conv = nn.Conv2d(condition_channels, base_channels, kernel_size=3, padding=1)

        self.block1 = TimeNAFBlock(base_channels, time_dim)
        self.block2 = TimeNAFBlock(base_channels, time_dim)

    def forward(self, condition, t_emb):
        x = self.in_conv(condition)
        x = self.block1(x, t_emb)
        x = self.block2(x, t_emb)
        return x

class DBCRControlNet(nn.Module):
    def __init__(self, sar_channels=2, base_channels=64, time_dim=256):
        super().__init__()
 
        self.sar_encoder = nn.Sequential(
            nn.Conv2d(sar_channels, base_channels, kernel_size=3, padding=1),
            LayerNorm2d(base_channels),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
        )
 
        self.down1 = DownBlockNAF(base_channels,     base_channels * 2, time_dim)
        self.down2 = DownBlockNAF(base_channels * 2, base_channels * 4, time_dim)
        self.down3 = DownBlockNAF(base_channels * 4, base_channels * 8, time_dim)
 
        self.mid1 = TimeNAFBlock(base_channels * 8, time_dim)
        self.mid2 = TimeNAFBlock(base_channels * 8, time_dim)
 
        self.zero_conv_skip1 = nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=1)
        self.zero_conv_skip2 = nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=1)
        self.zero_conv_skip3 = nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=1)
        self.zero_conv_mid   = nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=1)
 
        self._init_zero_convs()
 
    def _init_zero_convs(self):
        for layer in [
            self.zero_conv_skip1,
            self.zero_conv_skip2,
            self.zero_conv_skip3,
            self.zero_conv_mid,
        ]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
 
    def forward(self, sar, t_emb):
        x = self.sar_encoder(sar)
 
        x, skip1 = self.down1(x, t_emb)
        x, skip2 = self.down2(x, t_emb)
        x, skip3 = self.down3(x, t_emb)
 
        x = self.mid1(x, t_emb)
        x = self.mid2(x, t_emb)
 
        return {
            "skip1": self.zero_conv_skip1(skip1),
            "skip2": self.zero_conv_skip2(skip2),
            "skip3": self.zero_conv_skip3(skip3),
            "mid":   self.zero_conv_mid(x),
        }

class DBCRSimple(nn.Module):
    def __init__(
        self,
        image_channels=6,
        condition_channels=6,
        base_channels=64,
        time_dim=256,
        control_net = None
    ):
        super().__init__()

        self.time_mlp = TimeMLP(time_dim)
        if control_net:
            self.control_net = DBCRControlNet(base_channels=base_channels, time_dim=time_dim)

        else:
            self.control_net = None
        self.condition_encoder = ConditionEncoderNAF(
            condition_channels=condition_channels,
            base_channels=base_channels,
            time_dim=time_dim
        )

        self.init_conv = nn.Conv2d(
            image_channels + base_channels,
            base_channels,
            kernel_size=3,
            padding=1
        )

        self.down1 = DownBlockNAF(base_channels,     base_channels * 2, time_dim)
        self.down2 = DownBlockNAF(base_channels * 2, base_channels * 4, time_dim)
        self.down3 = DownBlockNAF(base_channels * 4, base_channels * 8, time_dim)

        self.mid1 = TimeNAFBlock(base_channels * 8, time_dim)
        self.mid2 = TimeNAFBlock(base_channels * 8, time_dim)

        self.up3 = UpBlockNAF(base_channels * 8, base_channels * 8, base_channels * 4, time_dim)
        self.up2 = UpBlockNAF(base_channels * 4, base_channels * 4, base_channels * 2, time_dim)
        self.up1 = UpBlockNAF(base_channels * 2, base_channels * 2, base_channels,     time_dim)

        self.out = nn.Conv2d(base_channels, image_channels, kernel_size=3, padding=1)

    def freeze_unet(self):
        for name, param in self.named_parameters():
            if name.startswith("control_net."):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, x_t, t, s2_cloudy, sar=None):
        t_emb = self.time_mlp(t)

        condition_features = self.condition_encoder(s2_cloudy, t_emb)

        x = torch.cat([x_t, condition_features], dim=1)
        x = self.init_conv(x)

        x, skip1 = self.down1(x, t_emb)
        x, skip2 = self.down2(x, t_emb)
        x, skip3 = self.down3(x, t_emb)

        x = self.mid1(x, t_emb)
        x = self.mid2(x, t_emb)

        if self.control_net is not None and sar is not None:
            residuals = self.control_net(sar, t_emb)
            skip1 = skip1 + residuals["skip1"]
            skip2 = skip2 + residuals["skip2"]
            skip3 = skip3 + residuals["skip3"]
            x     = x     + residuals["mid"]

        x = self.up3(x, skip3, t_emb)
        x = self.up2(x, skip2, t_emb)
        x = self.up1(x, skip1, t_emb)

        pred_clean = self.out(x)

        return pred_clean