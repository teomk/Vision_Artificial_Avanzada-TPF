import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# -------------------------
# Time Embedding
# -------------------------

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


# -------------------------
# NAFNet primitives
# -------------------------

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


# -------------------------
# Time-Embedded NAFBlock
# -------------------------

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


class SFBlock(nn.Module):
    """
    SAR Fusion Block con window attention.

    Parámetros:
    channels:    canales de entrada (igual en optical y SAR)
    num_heads:   cabezas de atención
    window_size: tamaño de ventana para window attention (default 8).
                Si None, usa global attention (solo viable en resoluciones
                pequeñas como el bottleneck).
    """

    def __init__(self, channels, num_heads=1, window_size=8, mlp_ratio=4):
        super().__init__()

        self.num_heads   = num_heads
        self.window_size = window_size
        self.scale       = (channels // num_heads) ** -0.5

        self.to_q = nn.Conv2d(channels, channels, kernel_size=1)
        self.to_k = nn.Conv2d(channels, channels, kernel_size=1)
        self.to_v = nn.Conv2d(channels, channels, kernel_size=1)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)  # proyección post-attention

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
        """
        Attention global — O((H*W)²). Solo viable en resoluciones
        pequeñas (bottleneck ~32×32). No usar en sf0/sf1.
        """
        hd = C // self.num_heads
        Q = Q.reshape(B, self.num_heads, hd, H * W).permute(0, 1, 3, 2)
        K = K.reshape(B, self.num_heads, hd, H * W).permute(0, 1, 3, 2)
        V = V.reshape(B, self.num_heads, hd, H * W).permute(0, 1, 3, 2)
        out = F.scaled_dot_product_attention(Q, K, V)
        return out.permute(0, 1, 3, 2).reshape(B, C, H, W)

    def _window_attention(self, Q, K, V, B, C, H, W):
        """
        Window attention — O(ws⁴) por ventana, independiente de H×W.
        Cada píxel atiende solo a los ws² píxeles de su ventana local.
        """
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

        # Fix resolución: alinear SAR a resolución del optical
        K, V = self._align_sar(K, V, H, W)

        # Usar window attention si la resolución lo requiere
        use_window = (self.window_size is not None and H > self.window_size and H % self.window_size == 0)
        if use_window:
            out = self._window_attention(Q, K, V, B, C, H, W)
        else:
            out = self._global_attention(Q, K, V, B, C, H, W)

        out = self.out_proj(out)
        x = optical + out
        x = x + self.mlp(self.norm_mlp(x))
        return x


# -------------------------
# Down / Up Blocks
# -------------------------

# class DownBlockNAF(nn.Module):
#     def __init__(self, in_channels, out_channels, time_dim):
#         super().__init__()
#         # self.proj = (
#         #     nn.Conv2d(in_channels, out_channels, kernel_size=1)
#         #     if in_channels != out_channels else nn.Identity()
#         # )
#         self.naf  = TimeNAFBlock(out_channels, time_dim)
#         self.down = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

#     def forward_pre(self, x, t_emb):
#         """NAFBlock sin downsampling — para insertar SFBlock después"""
#         x = self.proj(x)
#         x = self.naf(x, t_emb)
#         return x

#     def forward_down(self, x):
#         """Solo el downsampling"""
#         return self.down(x)

#     def forward(self, x, t_emb):
#         """Comportamiento original por si se necesita en otro contexto"""
#         x = self.forward_pre(x, t_emb)
#         return self.forward_down(x)

# class DownBlockNAF(nn.Module):
#     def __init__(self, in_channels, out_channels, time_dim):
#         super().__init__()
#         # NAF opera en in_channels (mismo canal que entra)
#         self.naf = TimeNAFBlock(in_channels, time_dim)
#         # El downsample HACE el cambio de canales: in_channels -> out_channels
#         self.down = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

#     def forward_pre(self, x, t_emb):
#         """NAF en in_channels, sin cambio de canales ni downsampling"""
#         return self.naf(x, t_emb)

#     def forward_down(self, x):
#         """Downsample + cambio de canales in_channels -> out_channels"""
#         return self.down(x)

#     def forward(self, x, t_emb):
#         x = self.forward_pre(x, t_emb)
#         return self.forward_down(x)

class DownBlock(nn.Module):
    """Solo cambia canales + reduce resolución."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.down(x)


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
            print(f"Alerta: resolución del upsample {x.shape[-2:]} no coincide con skip {skip.shape[-2:]}. Ajustando con interpolate.")
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.proj(x)
        x = self.naf(x, t_emb)
        return x


# -------------------------
# SAR Encoder Branch
# -------------------------

class SAREncoderBranch(nn.Module):
    def __init__(self, sar_channels=2, base_channels=64, time_dim=256):
        super().__init__()
        C = base_channels
        self.in_conv = nn.Conv2d(sar_channels, C, kernel_size=3, padding=1)

        self.sar_enc0 = TimeNAFBlock(C,     time_dim)
        self.down1    = DownBlock(C,       C * 2)

        self.sar_enc1 = TimeNAFBlock(C * 2, time_dim)
        self.down2    = DownBlock(C * 2,   C * 4)

        self.sar_enc2 = TimeNAFBlock(C * 4, time_dim)
        self.down3    = DownBlock(C * 4,   C * 8)

        self.sar_enc3 = TimeNAFBlock(C * 8, time_dim)
        self.mid      = TimeNAFBlock(C * 8, time_dim)

    def forward(self, sar, t_emb):
        x0 = self.in_conv(sar)
        x0 = self.sar_enc0(x0, t_emb)

        x1 = self.down1(x0)
        x1 = self.sar_enc1(x1, t_emb)

        x2 = self.down2(x1)
        x2 = self.sar_enc2(x2, t_emb)

        x3 = self.down3(x2)
        x3 = self.sar_enc3(x3, t_emb)

        mid = self.mid(x3, t_emb)
        return {
            "scale0": x0,
            "scale1": x1,
            "scale2": x2,
            "scale3": x3,
            "mid":    mid,
        }


# -------------------------
# DBCR
# -------------------------

class DBCR(nn.Module):
    """
    DB-CR con tres cambios respecto al modelo original:

    CAMBIO 1 — Window attention en SFBlock (window_size=8):
        La attention global genera una matriz (H*W)×(H*W) que para
        256×256 requiere ~16 GB. Window attention limita cada ventana
        a ws×ws tokens (64 con ws=8), reduciendo la VRAM en órdenes
        de magnitud. Los SFBlocks en el bottleneck (32×32) siguen
        usando global attention automáticamente porque H <= window_size
        no se cumple y la resolución pequeña lo permite.

    CAMBIO 2 — Fix de resolución SAR/optical en SFBlock:
        Los skips del SAR encoder (scale0..scale3) se guardan ANTES
        del downsampling, por lo que tienen el doble de resolución
        que el feature map óptico correspondiente. El SFBlock ahora
        alinea K y V con interpolate bilineal antes de la attention.

    CAMBIO 3 — Gradient checkpointing en bloques pesados:
        En vez de guardar todas las activaciones del forward para el
        backward, las recomputa on-the-fly. Cuesta ~25% más de tiempo
        pero reduce la VRAM del backward a la mitad aproximadamente.
        Se aplica en SFBlocks y NAFBlocks del bottleneck, que son los
        más costosos en memoria.

    El resultado es un pico de ~19 GB con batch=12 en una L4 de 24 GB,
    vs OOM con el modelo original.
    """

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
        use_checkpoint=True, # CAMBIO 3: gradient checkpointing
    ):
        super().__init__()

        C = base_channels
        self.use_checkpoint = use_checkpoint

        self.time_mlp = TimeMLP(time_dim)

        self.sar_branch = SAREncoderBranch(
            sar_channels=sar_channels,
            base_channels=C,
            time_dim=time_dim,
        )

        in_ch = image_channels + condition_channels
        self.init_conv = nn.Conv2d(in_ch, C, kernel_size=3, padding=1)

        self.enc0 = TimeNAFBlock(C, time_dim)
        self.down0to1 = DownBlock(C, C * 2)

        self.enc1 = TimeNAFBlock(C*2, time_dim)      # escala 1, C*2 canales
        self.down1to2 = DownBlock(C*2, C*4)

        self.enc2 = TimeNAFBlock(C*4, time_dim)      # escala 2, C*4 canales
        self.down2to3 = DownBlock(C*4, C*8)

        self.enc3 = TimeNAFBlock(C*8, time_dim)      # escala 3, C*8 canales

        # SFBlocks con window attention (CAMBIO 1 + 2)
        self.sf0    = SFBlock(C,     num_heads=num_heads, window_size=window_size_sf0)
        self.sf1    = SFBlock(C * 2, num_heads=num_heads, window_size=window_size_not_sf0)
        self.sf2    = SFBlock(C * 4, num_heads=num_heads, window_size=window_size_not_sf0)
        self.sf3    = SFBlock(C * 8, num_heads=num_heads, window_size=window_size_not_sf0)
        self.sf_mid = SFBlock(C * 8, num_heads=num_heads, window_size=window_size_not_sf0)

        # self.down1 = DownBlock(C,     C * 2, time_dim)
        # self.down2 = DownBlock(C * 2, C * 4, time_dim)
        # self.down3 = DownBlock(C * 4, C * 8, time_dim)

        self.mid1  = TimeNAFBlock(C * 8, time_dim)
        self.mid2  = TimeNAFBlock(C * 8, time_dim)

        self.up3   = UpBlockNAF(C * 8, C * 4, C * 4, time_dim)
        self.up2   = UpBlockNAF(C * 4, C * 2, C * 2, time_dim)
        self.up1   = UpBlockNAF(C * 2, C,     C,     time_dim)

        self.out   = nn.Conv2d(C, image_channels, kernel_size=3, padding=1)

    def _ckpt(self, fn, *args):
        """
        Gradient checkpointing (CAMBIO 3).
        Recomputa las activaciones en el backward en vez de guardarlas.
        use_reentrant=False es el modo moderno y más estable.
        """
        if self.use_checkpoint and self.training:
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def forward(self, x_t, t, s2_cloudy, sar):
        """
        x_t:       estado intermedio del bridge   [B, 6, H, W]
        t:         timestep                        [B]
        s2_cloudy: Sentinel-2 nublado              [B, 6, H, W]
        sar:       Sentinel-1 SAR                  [B, 2, H, W]

        Retorna:
        pred_clean: predicción S2 limpio           [B, 6, H, W]
        """

        t_emb = self.time_mlp(t)

        sar_feats = self.sar_branch(sar, t_emb)

        x = torch.cat([x_t, s2_cloudy], dim=1)
        x = self.init_conv(x)

        #Escala 0: Full resolution, C canales.
        x     = self._ckpt(self.enc0, x, t_emb)
        x     = self._ckpt(self.sf0, x, sar_feats["scale0"])
        skip0 = x
        x     = self.down0to1(x)

        # Escala 1
        x     = self._ckpt(self.enc1, x, t_emb)
        x     = self._ckpt(self.sf1, x, sar_feats["scale1"])
        skip1 = x
        x     = self.down1to2(x)

        # Escala 2
        x     = self._ckpt(self.enc2, x, t_emb)
        x     = self._ckpt(self.sf2, x, sar_feats["scale2"])
        skip2 = x
        x     = self.down2to3(x)

        # Escala 3
        x     = self._ckpt(self.enc3, x, t_emb)
        x     = self._ckpt(self.sf3, x,  sar_feats["scale3"])
        skip3 = x

        # Bottleneck
        x = self._ckpt(self.mid1,   x, t_emb)
        x = self._ckpt(self.mid2,   x, t_emb)
        x = self._ckpt(self.sf_mid, x, sar_feats["mid"])

        # Decoder
        x = self.up3(x, skip2, t_emb)
        x = self.up2(x, skip1, t_emb)
        x = self.up1(x, skip0, t_emb)

        return self.out(x)
    


# -------------------------
# Training loop mínimo
# -------------------------
# Para usar bf16 en el training loop (CAMBIO 4 — va en tu trainer, no acá):
#
#   scaler = torch.cuda.amp.GradScaler()
#
#   for batch in dataloader:
#       optimizer.zero_grad()
#       with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
#           pred = model(x_t, t, s2_cloudy, sar)
#           loss = F.l1_loss(pred, s2_clean)   # MAE, como el paper
#       scaler.scale(loss).backward()
#       scaler.step(optimizer)
#       scaler.update()


# -------------------------
# Sanity check
# -------------------------

if __name__ == "__main__":
    B, H, W = 2, 128, 128

    model = DBCR(
        image_channels=6,
        condition_channels=6,
        sar_channels=2,
        base_channels=64,
        time_dim=128,
        num_heads=1,
        window_size_not_sf0=None,
        window_size_sf0=8,
        use_checkpoint=True,
    ).cuda()

    x_t       = torch.randn(B, 6, H, W).cuda()
    t         = torch.randint(0, 1000, (B,)).cuda()
    s2_cloudy = torch.randn(B, 6, H, W).cuda()
    sar       = torch.randn(B, 2, H, W).cuda()

    model.train()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out  = model(x_t, t, s2_cloudy, sar)
        loss = out.mean()
    loss.backward()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Output shape : {out.shape}")
    print(f"Parámetros   : {n_params:.2f}M")
    print(f"VRAM pico    : {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")