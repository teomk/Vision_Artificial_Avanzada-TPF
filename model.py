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
# Blocks
# -------------------------

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


# -------------------------
# Condition Encoder
# -------------------------

class ConditionEncoder(nn.Module):
    """
    Recibe S2_cloudy + cloud_mask (4 canales por defecto).
    Produce features del mismo tamaño espacial que la entrada,
    con base_channels canales.
    """
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


# -------------------------
# ControlNet
# -------------------------

class ControlNet(nn.Module):
    """
    Módulo ControlNet para condicionar con SAR.

    La idea es:
    - Tiene su propio encoder (espejo del U-Net base).
    - Procesa la imagen SAR y produce "residuals" en cada escala.
    - Esos residuals se SUMAN a los skips del U-Net base.
    - El U-Net base queda CONGELADO, solo se entrena el ControlNet.

    sar_channels: cantidad de bandas SAR (Sentinel-1 tiene 2: VV y VH)
    base_channels: tiene que coincidir con el U-Net base
    time_dim: tiene que coincidir con el U-Net base
    """
    def __init__(self, sar_channels=2, base_channels=64, time_dim=256):
        super().__init__()

        # Proyecta SAR al espacio de features del U-Net
        self.sar_encoder = nn.Sequential(
            nn.Conv2d(sar_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU()
        )

        # Espejo del encoder del U-Net base
        self.down1 = DownBlock(base_channels,     base_channels * 2, time_dim)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, time_dim)
        self.down3 = DownBlock(base_channels * 4, base_channels * 8, time_dim)

        self.mid = ResBlock(base_channels * 8, base_channels * 8, time_dim)

        # Proyecciones "zero conv": inicializadas en cero para no romper
        # el U-Net base al arrancar el entrenamiento de la etapa 2.
        # Al inicio, los residuals son cero → el U-Net se comporta igual que antes.
        self.zero_conv_skip1 = nn.Conv2d(base_channels * 2, base_channels * 2, 1)
        self.zero_conv_skip2 = nn.Conv2d(base_channels * 4, base_channels * 4, 1)
        self.zero_conv_skip3 = nn.Conv2d(base_channels * 8, base_channels * 8, 1)
        self.zero_conv_mid   = nn.Conv2d(base_channels * 8, base_channels * 8, 1)

        self._init_zero_convs()

    def _init_zero_convs(self):
        """
        Inicializa los zero convs en cero.
        Esto es crítico: al inicio del entrenamiento de la etapa 2
        los residuals son exactamente 0, por lo que el U-Net base
        se comporta idéntico a como quedó en la etapa 1.
        El ControlNet aprende a partir de ese punto sin "resetear" lo aprendido.
        """
        for layer in [self.zero_conv_skip1, self.zero_conv_skip2,
                      self.zero_conv_skip3, self.zero_conv_mid]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, sar, t_emb):
        """
        sar:   imagen SAR.  Shape: [B, sar_channels, H, W]
        t_emb: time embedding ya calculado por el U-Net base. Shape: [B, time_dim]

        Devuelve un dict con los residuals para cada escala.
        """
        x = self.sar_encoder(sar)

        x, skip1 = self.down1(x, t_emb)
        x, skip2 = self.down2(x, t_emb)
        x, skip3 = self.down3(x, t_emb)

        x = self.mid(x, t_emb)

        return {
            "skip1": self.zero_conv_skip1(skip1),
            "skip2": self.zero_conv_skip2(skip2),
            "skip3": self.zero_conv_skip3(skip3),
            "mid":   self.zero_conv_mid(x),
        }


# -------------------------
# Conditional DDPM U-Net
# -------------------------

class ConditionalDDPMUNet(nn.Module):
    """
    U-Net condicional para DDPM con soporte opcional de ControlNet (SAR).

    Etapa 1 (sin SAR):
        model = ConditionalDDPMUNet()
        noise_pred = model(x_t, t, s2_cloudy, cloud_mask)

    Etapa 2 (con SAR, ControlNet):
        controlnet = ControlNet()
        model = ConditionalDDPMUNet(controlnet=controlnet)

        # Congelar U-Net, solo entrenar ControlNet
        model.freeze_unet()

        noise_pred = model(x_t, t, s2_cloudy, cloud_mask, sar=sar_image)
    """
    def __init__(
        self,
        image_channels=3,
        condition_channels=4,   # 3 bandas S2_cloudy + 1 máscara
        base_channels=64,
        time_dim=256,
        controlnet=None         # None en etapa 1, instancia de ControlNet en etapa 2
    ):
        super().__init__()

        self.controlnet = controlnet

        self.time_mlp = TimeMLP(time_dim)

        self.condition_encoder = ConditionEncoder(
            condition_channels=condition_channels,
            base_channels=base_channels
        )

        self.init_conv = nn.Conv2d(
            image_channels + base_channels,
            base_channels,
            kernel_size=3,
            padding=1
        )

        self.down1 = DownBlock(base_channels,     base_channels * 2, time_dim)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, time_dim)
        self.down3 = DownBlock(base_channels * 4, base_channels * 8, time_dim)

        self.mid1 = ResBlock(base_channels * 8, base_channels * 8, time_dim)
        self.mid2 = ResBlock(base_channels * 8, base_channels * 8, time_dim)

        self.up3 = UpBlock(base_channels * 8, base_channels * 8, base_channels * 4, time_dim)
        self.up2 = UpBlock(base_channels * 4, base_channels * 4, base_channels * 2, time_dim)
        self.up1 = UpBlock(base_channels * 2, base_channels * 2, base_channels,     time_dim)

        self.out = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, image_channels, 3, padding=1)
        )

    def freeze_unet(self):
        """
        Congela todos los parámetros del U-Net base.
        Llamar antes de entrenar la etapa 2 (ControlNet).
        El ControlNet en sí NO se congela porque es un módulo separado.
        """
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_unet(self):
        """
        Descongela el U-Net base (por si querés fine-tunear todo junto después).
        """
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x_t, t, s2_cloudy, cloud_mask, sar=None):
        """
        x_t:        imagen limpia con ruido.       Shape: [B, C, H, W]
        t:          timestep de difusión.          Shape: [B]
        s2_cloudy:  imagen S2 nubosa.              Shape: [B, 3, H, W]
        cloud_mask: máscara de nubes.              Shape: [B, 1, H, W]
        sar:        imagen SAR (opcional).         Shape: [B, 2, H, W]
                    Si es None, se comporta como etapa 1.
        """

        # --- Time embedding (compartido con ControlNet) ---
        t_emb = self.time_mlp(t)

        # --- Condición base: S2_cloudy + máscara ---
        condition = torch.cat([s2_cloudy, cloud_mask], dim=1)
        condition_features = self.condition_encoder(condition)

        # --- Entrada al U-Net ---
        x = torch.cat([x_t, condition_features], dim=1)
        x = self.init_conv(x)

        # --- Encoder ---
        x, skip1 = self.down1(x, t_emb)
        x, skip2 = self.down2(x, t_emb)
        x, skip3 = self.down3(x, t_emb)

        # --- Bottleneck ---
        x = self.mid1(x, t_emb)
        x = self.mid2(x, t_emb)

        # --- ControlNet: suma residuals si hay SAR ---
        if sar is not None and self.controlnet is not None:
            residuals = self.controlnet(sar, t_emb)

            # Los residuals se suman a los skips ANTES de que los use el decoder.
            # Así el SAR influye en todas las escalas de la reconstrucción.
            skip1 = skip1 + residuals["skip1"]
            skip2 = skip2 + residuals["skip2"]
            skip3 = skip3 + residuals["skip3"]
            x     = x     + residuals["mid"]

        # --- Decoder ---
        x = self.up3(x, skip3, t_emb)
        x = self.up2(x, skip2, t_emb)
        x = self.up1(x, skip1, t_emb)

        noise_pred = self.out(x)

        return noise_pred


# # -------------------------
# # Ejemplo de uso
# # -------------------------

# if __name__ == "__main__":
#     B, H, W = 2, 64, 64

#     # ---------- Etapa 1: sin SAR ----------
#     model_etapa1 = ConditionalDDPMUNet()

#     x_t       = torch.randn(B, 3, H, W)
#     t         = torch.randint(0, 1000, (B,))
#     s2_cloudy = torch.randn(B, 3, H, W)
#     mask      = torch.randn(B, 1, H, W)

#     out1 = model_etapa1(x_t, t, s2_cloudy, mask)
#     print("Etapa 1 output:", out1.shape)  # [2, 3, 64, 64]

#     # ---------- Etapa 2: con SAR + ControlNet ----------
#     controlnet    = ControlNet(sar_channels=2)
#     model_etapa2  = ConditionalDDPMUNet(controlnet=controlnet)

#     # Cargar pesos de etapa 1
#     # model_etapa2.load_state_dict(torch.load("etapa1.pth"), strict=False)

#     # Congelar U-Net, solo entrenar ControlNet
#     model_etapa2.freeze_unet()

#     sar = torch.randn(B, 2, H, W)
#     out2 = model_etapa2(x_t, t, s2_cloudy, mask, sar=sar)
#     print("Etapa 2 output:", out2.shape)  # [2, 3, 64, 64]

#     # Verificar que el U-Net está congelado y el ControlNet no
#     unet_params     = sum(p.numel() for p in model_etapa2.parameters() if p.requires_grad)
#     control_params  = sum(p.numel() for p in controlnet.parameters()   if p.requires_grad)
#     print(f"Params entrenables U-Net:    {unet_params}")      # debe ser 0
#     print(f"Params entrenables ControlNet: {control_params}") # debe ser > 0