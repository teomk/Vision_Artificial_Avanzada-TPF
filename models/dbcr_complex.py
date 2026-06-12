import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """
    NAFBlock con modulación temporal mediante time embedding.
    Idéntico al del modelo original.
    """

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


# -------------------------
# SFBlock: SAR Fusion Block
# -------------------------

class SFBlock(nn.Module):
    """
    SAR Fusion Block (SFBlock) — Figura 3(b) del paper DB-CR.

    Implementa cross-modal attention entre los features ópticos y SAR:
      - Query  ← features ópticos  (rama U-Net)
      - Key    ← features SAR      (rama SAR encoder)
      - Value  ← features SAR      (rama SAR encoder)

    El output enriquece los features ópticos con información estructural
    del SAR, y se devuelve sumado al residual óptico original.

    Notas de implementación:
      - La atención se calcula en el espacio espacial aplanado (H*W tokens).
      - num_heads=1 por defecto; se puede aumentar para modelos más grandes.
      - Una Conv2d 3×3 final mezcla la salida antes del residual,
        tal como muestra la Figura 3(b).
    """

    def __init__(self, channels, num_heads=1):
        super().__init__()

        self.num_heads = num_heads
        self.scale     = (channels // num_heads) ** -0.5

        # Proyecciones lineales para Q, K, V
        self.to_q = nn.Conv2d(channels, channels, kernel_size=1)
        self.to_k = nn.Conv2d(channels, channels, kernel_size=1)
        self.to_v = nn.Conv2d(channels, channels, kernel_size=1)

        # Mezcla final: Conv 3×3 tal como indica la figura del paper
        self.out_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        self.norm_opt = LayerNorm2d(channels)
        self.norm_sar = LayerNorm2d(channels)

    def forward(self, optical, sar):
        """
        optical: [B, C, H, W]  — features de la rama óptica (U-Net)
        sar:     [B, C, H, W]  — features de la rama SAR (misma escala)

        Retorna:
        optical enriquecido: [B, C, H, W]
        """
        B, C, H, W = optical.shape

        opt_n = self.norm_opt(optical)
        sar_n = self.norm_sar(sar)

        # Q desde óptico, K/V desde SAR
        Q = self.to_q(opt_n)  # [B, C, H, W]
        K = self.to_k(sar_n)  # [B, C, H, W]
        V = self.to_v(sar_n)  # [B, C, H, W]

        # Reshape para multi-head attention: [B, heads, H*W, C//heads]
        head_dim = C // self.num_heads
        Q = Q.reshape(B, self.num_heads, head_dim, H * W).permute(0, 1, 3, 2)
        K = K.reshape(B, self.num_heads, head_dim, H * W).permute(0, 1, 3, 2)
        V = V.reshape(B, self.num_heads, head_dim, H * W).permute(0, 1, 3, 2)

        # # Scaled dot-product attention
        # attn = (Q @ K.transpose(-2, -1)) * self.scale   # [B, heads, H*W, H*W]
        # attn = attn.softmax(dim=-1)

        # out = attn @ V                                   # [B, heads, H*W, head_dim]
        # out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)

        out = F.scaled_dot_product_attention(Q, K, V)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)

        # Conv 3×3 de salida + residual óptico
        out = self.out_conv(out)
        return optical + out


# -------------------------
# Down / Up Blocks
# -------------------------

class DownBlockNAF(nn.Module):
    """
    Baja resolución a la mitad.
    Internamente: NAFBlock → guarda skip → Conv stride-2.
    """

    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()

        self.proj = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )
        self.naf  = TimeNAFBlock(out_channels, time_dim)
        self.down = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x, t_emb):
        x    = self.proj(x)
        x    = self.naf(x, t_emb)
        skip = x
        x    = self.down(x)
        return x, skip


class UpBlockNAF(nn.Module):
    """
    Sube resolución al doble y fusiona skip connection.
    """

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
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.proj(x)
        x = self.naf(x, t_emb)
        return x


# -------------------------
# SAR Encoder Branch
# -------------------------

class SAREncoderBranch(nn.Module):
    """
    Rama paralela de extracción de features SAR.

    Espejo del encoder óptico: mismas escalas, mismos tamaños de canal.
    Produce features a 4 escalas (3 downsampling + bottleneck) que luego
    se fusionan con la rama óptica mediante SFBlocks.

    A diferencia del ControlNet del modelo original, esta rama NO usa
    zero-convs ni residuals sumados. Los features van directamente a los
    SFBlocks como Key/Value de la cross-attention.

    No comparte pesos con la rama óptica: aprende representaciones
    SAR-específicas desde cero.
    """

    def __init__(self, sar_channels=2, base_channels=64, time_dim=256):
        super().__init__()

        C = base_channels

        # Proyección inicial SAR → espacio de features
        self.in_conv = nn.Conv2d(sar_channels, C, kernel_size=3, padding=1)

        # Encoder espejo (mismo número de escalas que la rama óptica)
        self.down1 = DownBlockNAF(C,     C * 2, time_dim)
        self.down2 = DownBlockNAF(C * 2, C * 4, time_dim)
        self.down3 = DownBlockNAF(C * 4, C * 8, time_dim)

        # Bottleneck SAR
        self.mid = TimeNAFBlock(C * 8, time_dim)

    def forward(self, sar, t_emb):
        """
        Retorna dict con features a cada escala:
          'scale0': resolución full   [B, C,   H,   W]
          'scale1': 1/2               [B, 2C,  H/2, W/2]
          'scale2': 1/4               [B, 4C,  H/4, W/4]
          'scale3': 1/8               [B, 8C,  H/8, W/8]
          'mid':    1/8 (bottleneck)  [B, 8C,  H/8, W/8]
        """
        x0 = self.in_conv(sar)          # full res, C canales

        x, skip1 = self.down1(x0, t_emb)
        x, skip2 = self.down2(x,  t_emb)
        x, skip3 = self.down3(x,  t_emb)

        mid = self.mid(x, t_emb)

        return {
            "scale0": x0,
            "scale1": skip1,
            "scale2": skip2,
            "scale3": skip3,
            "mid":    mid,
        }


# -------------------------
# DBCR — Paper Architecture
# -------------------------

class DBCR(nn.Module):
    """
    Reimplementación de DB-CR (arxiv 2504.03607) adaptada a 6 bandas S2.

    Arquitectura:
    ┌─────────────────────────────────────────────────┐
    │  Entrada: cat(x_t, s2_cloudy) → [B, 12, H, W]  │
    │           sar                 → [B,  2, H, W]   │
    └─────────────────────────────────────────────────┘
                        ↓
         ┌──────────────────────────┐
         │  SAR Encoder Branch      │  (features a 4 escalas)
         └──────────────────────────┘
                        ↕  SFBlock (cross-modal attention en cada escala)
         ┌──────────────────────────┐
         │  U-Net Óptico            │
         │  Encoder: 3× DownBlock   │
         │  Bottleneck: 2× NAFBlock │
         │  Decoder: 3× UpBlock     │
         └──────────────────────────┘
                        ↓
               Conv 3×3 → pred_clean [B, 6, H, W]

    Diferencias clave vs DBCRSimple (modelo propio):
      1. s2_cloudy se concatena crudo con x_t (sin ConditionEncoderNAF)
      2. SAR se fusiona por cross-attention en cada escala (no ControlNet/suma)
      3. SFBlock actúa ANTES de los skip connections del decoder
      4. La rama SAR y la óptica son paralelas desde el inicio

    Parámetros:
      image_channels:     canales de x_t y s2_clean   (default 6)
      condition_channels: canales de s2_cloudy         (default 6)
      sar_channels:       canales SAR                  (default 2)
      base_channels:      canales base del U-Net        (default 64)
      time_dim:           dimensión del time embedding  (default 256)
      num_heads:          cabezas de atención en SFBlock (default 1)
    """

    def __init__(
        self,
        image_channels=6,
        condition_channels=6,
        sar_channels=2,
        base_channels=64,
        time_dim=256,
        num_heads=1,
    ):
        super().__init__()

        C = base_channels

        # Time embedding compartido entre ambas ramas
        self.time_mlp = TimeMLP(time_dim)

        # --- Rama SAR ---
        self.sar_branch = SAREncoderBranch(
            sar_channels=sar_channels,
            base_channels=C,
            time_dim=time_dim,
        )

        # --- Rama óptica: U-Net ---

        # Entrada: cat(x_t, s2_cloudy) sin encoder intermedio
        in_ch = image_channels + condition_channels   # 6 + 6 = 12
        self.init_conv = nn.Conv2d(in_ch, C, kernel_size=3, padding=1)

        # SFBlock en resolución full (escala 0, antes del primer down)
        self.sf0 = SFBlock(C, num_heads=num_heads)

        # Encoder
        self.down1 = DownBlockNAF(C,     C * 2, time_dim)
        self.down2 = DownBlockNAF(C * 2, C * 4, time_dim)
        self.down3 = DownBlockNAF(C * 4, C * 8, time_dim)

        # SFBlocks post-encoder (enriquecen los skips antes del decoder)
        self.sf1 = SFBlock(C * 2, num_heads=num_heads)
        self.sf2 = SFBlock(C * 4, num_heads=num_heads)
        self.sf3 = SFBlock(C * 8, num_heads=num_heads)

        # Bottleneck
        self.mid1 = TimeNAFBlock(C * 8, time_dim)
        self.mid2 = TimeNAFBlock(C * 8, time_dim)

        # SFBlock en bottleneck
        self.sf_mid = SFBlock(C * 8, num_heads=num_heads)

        # Decoder
        self.up3 = UpBlockNAF(C * 8, C * 8, C * 4, time_dim)
        self.up2 = UpBlockNAF(C * 4, C * 4, C * 2, time_dim)
        self.up1 = UpBlockNAF(C * 2, C * 2, C,     time_dim)

        # Salida
        self.out = nn.Conv2d(C, image_channels, kernel_size=3, padding=1)

    def forward(self, x_t, t, s2_cloudy, sar):
        """
        x_t:       estado intermedio del bridge   [B, 6, H, W]
        t:         timestep                        [B]
        s2_cloudy: Sentinel-2 nublado              [B, 6, H, W]
        sar:       Sentinel-1 SAR                  [B, 2, H, W]

        Retorna:
        pred_clean: predicción S2 limpio           [B, 6, H, W]
        """

        # 1. Time embedding (compartido entre ambas ramas)
        t_emb = self.time_mlp(t)

        # 2. Extraer features SAR a todas las escalas
        sar_feats = self.sar_branch(sar, t_emb)

        # 3. Proyección inicial óptica: cat crudo sin encoder intermedio
        x = torch.cat([x_t, s2_cloudy], dim=1)   # [B, 12, H, W]
        x = self.init_conv(x)                     # [B,  C, H, W]

        # SFBlock escala 0 (resolución full)
        x = self.sf0(x, sar_feats["scale0"])

        # 4. Encoder óptico con fusión SAR en cada escala
        x, skip1 = self.down1(x, t_emb)
        skip1 = self.sf1(skip1, sar_feats["scale1"])   # enriquece skip

        x, skip2 = self.down2(x, t_emb)
        skip2 = self.sf2(skip2, sar_feats["scale2"])

        x, skip3 = self.down3(x, t_emb)
        skip3 = self.sf3(skip3, sar_feats["scale3"])

        # 5. Bottleneck con fusión SAR
        x = self.mid1(x, t_emb)
        x = self.mid2(x, t_emb)
        x = self.sf_mid(x, sar_feats["mid"])

        # 6. Decoder (usa skips ya enriquecidos con SAR)
        x = self.up3(x, skip3, t_emb)
        x = self.up2(x, skip2, t_emb)
        x = self.up1(x, skip1, t_emb)

        pred_clean = self.out(x)

        return pred_clean


# -------------------------
# Sanity check
# -------------------------

if __name__ == "__main__":
    # B, H, W = 2, 256, 256
    B, H, W = 2, 128, 128


    model = DBCR(
        image_channels=6,
        condition_channels=6,
        sar_channels=2,
        base_channels=64,
        time_dim=128,
        num_heads=1,
    )

    x_t       = torch.randn(B, 6, H, W)
    t         = torch.randint(0, 1000, (B,))
    s2_cloudy = torch.randn(B, 6, H, W)
    sar       = torch.randn(B, 2, H, W)

    with torch.no_grad():
        out = model(x_t, t, s2_cloudy, sar)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Output shape : {out.shape}")
    print(f"Parámetros   : {n_params:.2f}M")