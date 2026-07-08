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

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)

        self.time_proj = nn.Linear(time_dim, out_channels)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)

        time_emb = self.time_proj(t_emb)
        h = h + time_emb[:, :, None, None]

        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)

        return h + self.skip(x)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.res = ResBlock(in_channels, out_channels, time_dim)
        self.down = nn.Conv2d(out_channels, out_channels, 4, stride=2, padding=1)

    def forward(self, x, t_emb):
        x = self.res(x, t_emb)
        skip = x
        x = self.down(x)
        return x, skip

class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, time_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
        self.res = ResBlock(out_channels + skip_channels, out_channels, time_dim)

    def forward(self, x, skip, t_emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res(x, t_emb)
        return x

class ConditionEncoder(nn.Module):
    def __init__(self, condition_channels, base_channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(condition_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU()
        )

    def forward(self, condition):
        return self.net(condition)

class ControlNet(nn.Module):
    def __init__(self, sar_channels=2, base_channels=64, time_dim=256):
        super().__init__()

        self.sar_encoder = nn.Sequential(
            nn.Conv2d(sar_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU()
        )

        self.down1 = DownBlock(base_channels,     base_channels * 2, time_dim)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, time_dim)
        self.down3 = DownBlock(base_channels * 4, base_channels * 8, time_dim)

        self.mid = ResBlock(base_channels * 8, base_channels * 8, time_dim)

        self.zero_conv_skip1 = nn.Conv2d(base_channels * 2, base_channels * 2, 1)
        self.zero_conv_skip2 = nn.Conv2d(base_channels * 4, base_channels * 4, 1)
        self.zero_conv_skip3 = nn.Conv2d(base_channels * 8, base_channels * 8, 1)
        self.zero_conv_mid   = nn.Conv2d(base_channels * 8, base_channels * 8, 1)

        self._init_zero_convs()

    def _init_zero_convs(self):
        for layer in [self.zero_conv_skip1, self.zero_conv_skip2, self.zero_conv_skip3, self.zero_conv_mid]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, sar, t_emb):
        x = self.sar_encoder(sar)

        x, skip1 = self.down1(x, t_emb)
        x, skip2 = self.down2(x, t_emb)
        x, skip3 = self.down3(x, t_emb)

        x = self.mid(x, t_emb)

        return {"skip1": self.zero_conv_skip1(skip1),
                "skip2": self.zero_conv_skip2(skip2),
                "skip3": self.zero_conv_skip3(skip3),
                "mid":   self.zero_conv_mid(x),}

class ConditionalDDPMUNet(nn.Module):
    def __init__(self, image_channels=6, condition_channels=6, base_channels=64, time_dim=256, controlnet=None):
        super().__init__()

        self.controlnet = controlnet

        self.time_mlp = TimeMLP(time_dim)

        self.condition_encoder = ConditionEncoder(condition_channels=condition_channels, base_channels=base_channels)

        self.init_conv = nn.Conv2d(image_channels + base_channels, base_channels, kernel_size=3, padding=1)

        self.down1 = DownBlock(base_channels,     base_channels * 2, time_dim)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, time_dim)
        self.down3 = DownBlock(base_channels * 4, base_channels * 8, time_dim)

        self.mid1 = ResBlock(base_channels * 8, base_channels * 8, time_dim)
        self.mid2 = ResBlock(base_channels * 8, base_channels * 8, time_dim)

        self.up3 = UpBlock(base_channels * 8, base_channels * 8, base_channels * 4, time_dim)
        self.up2 = UpBlock(base_channels * 4, base_channels * 4, base_channels * 2, time_dim)
        self.up1 = UpBlock(base_channels * 2, base_channels * 2, base_channels,     time_dim)

        self.out = nn.Sequential(nn.GroupNorm(8, base_channels),
                                 nn.SiLU(),
                                 nn.Conv2d(base_channels, image_channels, 3, padding=1)
                                )

    def freeze_unet(self):
        for name, param in self.named_parameters():
            if self.controlnet is not None and name.startswith("controlnet."):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def unfreeze_unet(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x_t, t, s2_cloudy, sar=None):
        t_emb = self.time_mlp(t)

        condition_features = self.condition_encoder(s2_cloudy)

        x = torch.cat([x_t, condition_features], dim=1)
        x = self.init_conv(x)

        x, skip1 = self.down1(x, t_emb)
        x, skip2 = self.down2(x, t_emb)
        x, skip3 = self.down3(x, t_emb)

        x = self.mid1(x, t_emb)
        x = self.mid2(x, t_emb)

        if sar is not None and self.controlnet is not None:
            residuals = self.controlnet(sar, t_emb)
            skip1 = skip1 + residuals["skip1"]
            skip2 = skip2 + residuals["skip2"]
            skip3 = skip3 + residuals["skip3"]
            x     = x     + residuals["mid"]

        x = self.up3(x, skip3, t_emb)
        x = self.up2(x, skip2, t_emb)
        x = self.up1(x, skip1, t_emb)

        noise_pred = self.out(x)

        return noise_pred