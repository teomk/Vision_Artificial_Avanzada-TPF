import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

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
        self.bias   = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps    = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var  = x.var(dim=1,  keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class TimeNAFBlock(nn.Module):
    def __init__(self, channels, time_dim, dw_expand=2, ffn_expand=2):
        super().__init__()

        dw_ch  = channels * dw_expand
        ffn_ch = channels * ffn_expand

        self.norm1  = LayerNorm2d(channels)
        self.conv1  = nn.Conv2d(channels, dw_ch, kernel_size=1)
        self.dwconv = nn.Conv2d(dw_ch, dw_ch, kernel_size=3, padding=1, groups=dw_ch)
        self.sg     = SimpleGate()
        self.sca    = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_ch // 2, dw_ch // 2, kernel_size=1)
        )
        self.conv2  = nn.Conv2d(dw_ch // 2, channels, kernel_size=1)
        self.time_proj1 = nn.Linear(time_dim, channels)

        self.norm2  = LayerNorm2d(channels)
        self.conv3  = nn.Conv2d(channels, ffn_ch, kernel_size=1)
        self.conv4  = nn.Conv2d(ffn_ch // 2, channels, kernel_size=1)
        self.time_proj2 = nn.Linear(time_dim, channels)

        self.beta  = nn.Parameter(torch.zeros(1, channels, 1, 1))
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

class SARFBlock(nn.Module):
    def __init__(self, channels, num_heads=1, window_size=8, mlp_ratio=4):
        super().__init__()

        self.num_heads   = num_heads
        self.window_size = window_size
        self.scale       = (channels // num_heads) ** -0.5

        self.to_q = nn.Conv2d(channels, channels, kernel_size=1)
        self.to_k = nn.Conv2d(channels, channels, kernel_size=1)
        self.to_v = nn.Conv2d(channels, channels, kernel_size=1)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)

        self.norm_opt = LayerNorm2d(channels)
        self.norm_sar = LayerNorm2d(channels)
        self.norm_mlp = LayerNorm2d(channels)

        mlp_hidden = channels * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mlp_hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(mlp_hidden, channels, kernel_size=1)
        )

    def _align_sar(self, K, V, H, W):
        assert K.shape[-2:] == (H, W), f"SAR/optical resolution mismatch: {K.shape[-2:]} vs {(H,W)}"
        return K, V

    def _global_attention(self, Q, K, V, B, C, H, W):
        hd = C // self.num_heads
        Q = Q.reshape(B, self.num_heads, hd, H * W).permute(0, 1, 3, 2)
        K = K.reshape(B, self.num_heads, hd, H * W).permute(0, 1, 3, 2)
        V = V.reshape(B, self.num_heads, hd, H * W).permute(0, 1, 3, 2)
        out = F.scaled_dot_product_attention(Q, K, V)
        return out.permute(0, 1, 3, 2).reshape(B, C, H, W)

    def _window_attention(self, Q, K, V, B, C, H, W):
        ws     = self.window_size
        hd     = C // self.num_heads
        nH, nW = H // ws, W // ws

        def partition(t):
            # [B, C, H, W] → [B*nH*nW, heads, ws², hd]
            t = t.reshape(B, C, nH, ws, nW, ws)
            t = t.permute(0, 2, 4, 3, 5, 1).contiguous()  # [B, nH, nW, ws, ws, C]
            t = t.reshape(B * nH * nW, ws * ws, self.num_heads, hd)
            return t.permute(0, 2, 1, 3)                   # [B*nH*nW, heads, ws², hd]

        Q, K, V = partition(Q), partition(K), partition(V)
        out = F.scaled_dot_product_attention(Q, K, V)      # [B*nH*nW, heads, ws², hd]

        # Reconstruir imagen
        out = out.permute(0, 2, 1, 3).contiguous()         # [B*nH*nW, ws², heads, hd]
        out = out.reshape(B * nH * nW, ws, ws, C)
        out = out.reshape(B, nH, nW, ws, ws, C)
        out = out.permute(0, 5, 1, 3, 2, 4).contiguous()  # [B, C, nH, ws, nW, ws]
        return out.reshape(B, C, H, W)

    def forward(self, optical, sar):
        B, C, H, W = optical.shape

        opt_n = self.norm_opt(optical)
        sar_n = self.norm_sar(sar)

        Q = self.to_q(opt_n)
        K = self.to_k(sar_n)
        V = self.to_v(sar_n)

        K, V = self._align_sar(K, V, H, W)

        use_window = (self.window_size is not None and H > self.window_size and H % self.window_size == 0)
        if use_window:
            out = self._window_attention(Q, K, V, B, C, H, W)
        else:
            out = self._global_attention(Q, K, V, B, C, H, W)

        out = self.out_proj(out)
        x = optical + out
        x = x + self.mlp(self.norm_mlp(x))
        return x

class UpBlockNAF(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, time_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        in_ch_block = out_channels + skip_channels
        self.proj   = (
            nn.Conv2d(in_ch_block, out_channels, kernel_size=1)
            if in_ch_block != out_channels else nn.Identity()
        )
        self.naf = TimeNAFBlock(out_channels, time_dim)

    def forward(self, x, skip, t_emb):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            print(f"resolución del upsample {x.shape[-2:]} no coincide con skip {skip.shape[-2:]}. Ajustando con interpolate.")
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.proj(x)
        x = self.naf(x, t_emb)
        return x

class SAREncoderBranch(nn.Module):
    def __init__(self, sar_channels=2, base_channels=64, time_dim=128, include_encoder_4=False):
        super().__init__()
        C = base_channels
        self.has_enc4 = include_encoder_4

        self.in_conv = nn.Conv2d(sar_channels, C, kernel_size=3, padding=1)

        self.sar_enc1 = TimeNAFBlock(C,     time_dim)
        self.down1    = nn.Conv2d(C, C * 2, kernel_size=3, stride=2, padding=1)

        self.sar_enc2 = TimeNAFBlock(C * 2, time_dim)
        self.down2    = nn.Conv2d(C * 2, C * 4, kernel_size=3, stride=2, padding=1)

        self.sar_enc3 = TimeNAFBlock(C * 4, time_dim)
        self.down3    = nn.Conv2d(C * 4, C * 8, kernel_size=3, stride=2, padding=1)

        if include_encoder_4:
            self.sar_enc4 = TimeNAFBlock(C * 8, time_dim)
            self.down4    = nn.Conv2d(C * 8, C * 16, kernel_size=3, stride=2, padding=1)
            self.mid      = TimeNAFBlock(C * 16, time_dim)
        else:
            self.mid      = TimeNAFBlock(C * 8, time_dim)

    def forward(self, sar, t_emb):
        x1 = self.in_conv(sar)
        x1 = self.sar_enc1(x1, t_emb)

        x2 = self.down1(x1)
        x2 = self.sar_enc2(x2, t_emb)

        x3 = self.down2(x2)
        x3 = self.sar_enc3(x3, t_emb)

        x4 = self.down3(x3)
        x_mid = x4

        if self.has_enc4:
            x4 = self.sar_enc4(x4, t_emb)
            x_mid = self.down4(x4)
    
        mid = self.mid(x_mid, t_emb)
        return {
            "scale1": x1,
            "scale2": x2,
            "scale3": x3,
            "scale4": x4,
            "mid":    mid,
        }

class DownNAFSARF(nn.Module):
    def __init__(self, in_channels, down_channels, time_dim, num_heads=1, window_size=None):
        super().__init__()
        self.naf = TimeNAFBlock(in_channels, time_dim)
        self.sf  = SARFBlock(in_channels, num_heads=num_heads, window_size=window_size)
        self.down = nn.Conv2d(in_channels, down_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x, t_emb, sar_feat):
        x = self.naf(x, t_emb)
        x = self.sf(x, sar_feat)
        skip = x
        x = self.down(x)
        return x, skip

class DBCR(nn.Module):
    def __init__(
        self,
        image_channels=6,
        condition_channels=6,
        sar_channels=2,
        base_channels=64,
        time_dim=128,
        num_heads=1,
        window_size_not_sf0=None,        # sf1/sf2/sf3/sf_mid global
        window_size_sf0=8,       # solo sf0 con window
        use_checkpoint=True,
        include_encoder_4=False,
    ):
        super().__init__()

        C = base_channels
        self.has_enc4 = include_encoder_4
        self.use_checkpoint = use_checkpoint

        self.time_mlp = TimeMLP(time_dim)

        self.sar_branch = SAREncoderBranch(sar_channels=sar_channels, base_channels=C, time_dim=time_dim, include_encoder_4=include_encoder_4)

        in_ch = image_channels + condition_channels
        self.init_conv = nn.Conv2d(in_ch, C, kernel_size=3, padding=1)

        self.enc1 = DownNAFSARF(C, C*2, time_dim, num_heads=num_heads, window_size=window_size_sf0)
        self.enc2 = DownNAFSARF(C*2, C*4, time_dim, num_heads=num_heads, window_size=window_size_not_sf0)
        self.enc3 = DownNAFSARF(C*4, C*8, time_dim, num_heads=num_heads, window_size=window_size_not_sf0)

        mid_channels = C * 8
        if include_encoder_4:
            self.enc4 = DownNAFSARF(C*8, C*16, time_dim, num_heads=num_heads, window_size=window_size_not_sf0)
            mid_channels = C * 16

        self.mid1  = TimeNAFBlock(mid_channels, time_dim)
        self.sf_mid = SARFBlock(mid_channels, num_heads=num_heads, window_size=window_size_not_sf0)
        self.mid2  = TimeNAFBlock(mid_channels, time_dim)

        if include_encoder_4:
            self.up4   = UpBlockNAF(C * 16, C * 8, C * 8, time_dim)

        self.up3   = UpBlockNAF(C * 8, C * 4, C * 4, time_dim)
        self.up2   = UpBlockNAF(C * 4, C * 2, C * 2, time_dim)
        self.up1   = UpBlockNAF(C * 2, C,     C,     time_dim)

        self.out   = nn.Conv2d(C, image_channels, kernel_size=3, padding=1)

    def _ckpt(self, fn, *args):
        if self.use_checkpoint and self.training:
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def forward(self, x_t, t, s2_cloudy, sar):
        t_emb = self.time_mlp(t)

        sar_feats = self.sar_branch(sar, t_emb)

        x = torch.cat([x_t, s2_cloudy], dim=1)
        x = self.init_conv(x)

        x, skip1 = self._ckpt(self.enc1, x, t_emb, sar_feats["scale1"])

        x, skip2 = self._ckpt(self.enc2, x, t_emb, sar_feats["scale2"])

        x, skip3 = self._ckpt(self.enc3, x, t_emb, sar_feats["scale3"])

        if self.has_enc4:
          x, skip4 = self._ckpt(self.enc4, x, t_emb, sar_feats["scale4"])

        # Bottleneck
        x = self._ckpt(self.mid1,   x, t_emb)
        x = self._ckpt(self.sf_mid, x, sar_feats["mid"])
        x = self._ckpt(self.mid2,   x, t_emb)

        if self.has_enc4:
            x = self.up4(x, skip4, t_emb)

        x = self.up3(x, skip3, t_emb)
        x = self.up2(x, skip2, t_emb)
        x = self.up1(x, skip1, t_emb)

        return self.out(x)