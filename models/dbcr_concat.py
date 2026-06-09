"""
dbcr_sar_naf.py
---------------
Opción A: fusión SAR por concatenación simple en el ConditionEncoder.

Diferencia con DBCRNoSARNAF:
  - ConditionEncoder recibe [s2_cloudy (6ch) + s1 (2ch)] = 8 canales
  - forward() acepta s1 como argumento extra
  - Todo lo demás es idéntico
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reutilizamos todos los bloques del modelo No-SAR
from dbcr_no_sar_naf import (
    TimeMLP,
    LayerNorm2d,
    SimpleGate,
    TimeNAFBlock,
    NAFResBlock,
    DownBlockNAF,
    UpBlockNAF,
    ConditionEncoderNAF,
)


class DBCRSARNAFConcat(nn.Module):
    """
    Modelo con SAR - Fusión por concatenación.

    El SAR (2 canales S1) se concatena al S2_cloudy (6 canales)
    antes de entrar al ConditionEncoder → 8 canales de condición.

    Arquitectura idéntica a DBCRNoSARNAF, solo cambia condition_channels.
    """

    def __init__(
        self,
        image_channels=6,
        sar_channels=2,
        condition_channels=6,   # S2 channels (sin SAR)
        base_channels=64,
        time_dim=128,
    ):
        super().__init__()

        # Canales de condición = S2 + SAR
        fused_condition_channels = condition_channels + sar_channels  # 6 + 2 = 8

        self.time_mlp = TimeMLP(time_dim)

        self.condition_encoder = ConditionEncoderNAF(
            condition_channels=fused_condition_channels,  # 8 canales
            base_channels=base_channels,
            time_dim=time_dim,
        )

        self.init_conv = nn.Conv2d(
            image_channels + base_channels,
            base_channels,
            kernel_size=3,
            padding=1,
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

    def forward(self, x_t, t, s2_cloudy, s1):
        """
        x_t:      estado bridge.    [B, 6, H, W]
        t:        timestep.         [B]
        s2_cloudy: S2 nublado.      [B, 6, H, W]
        s1:       SAR Sentinel-1.   [B, 2, H, W]
        """
        t_emb = self.time_mlp(t)

        # Concatenar S2 + SAR como condición
        condition = torch.cat([s2_cloudy, s1], dim=1)  # [B, 8, H, W]
        condition_features = self.condition_encoder(condition, t_emb)

        x = torch.cat([x_t, condition_features], dim=1)
        x = self.init_conv(x)

        x, skip1 = self.down1(x, t_emb)
        x, skip2 = self.down2(x, t_emb)
        x, skip3 = self.down3(x, t_emb)

        x = self.mid1(x, t_emb)
        x = self.mid2(x, t_emb)

        x = self.up3(x, skip3, t_emb)
        x = self.up2(x, skip2, t_emb)
        x = self.up1(x, skip1, t_emb)

        pred_clean = self.out(x)

        return pred_clean