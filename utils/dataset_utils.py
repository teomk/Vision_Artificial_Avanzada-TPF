import torch

def unpack_batch(batch, sar_mode, device):
    """
    Retorna:
        s2_cloudy  [B, 6, H, W]
        s2_clean   [B, 6, H, W]
        condition  [B, 6 o 8, H, W]
        sar        [B, 2, H, W] o None
    """
    if sar_mode == "None":
        s2_cloudy, s2_clean = batch
        s2_cloudy = s2_cloudy.to(device)
        s2_clean  = s2_clean.to(device)
        condition = s2_cloudy
        sar       = None

    elif sar_mode == "Concat":
        s1, s2_cloudy, s2_clean = batch
        s1        = s1.to(device)
        s2_cloudy = s2_cloudy.to(device)
        s2_clean  = s2_clean.to(device)
        condition = torch.cat([s2_cloudy, s1], dim=1)   # [B, 8, H, W]
        sar       = None

    elif sar_mode == "ControlNet":
        s1, s2_cloudy, s2_clean = batch
        s1        = s1.to(device)
        s2_cloudy = s2_cloudy.to(device)
        s2_clean  = s2_clean.to(device)
        condition = s2_cloudy
        sar       = s1

    else:
        raise ValueError(f"sar_mode desconocido: '{sar_mode}'")

    return s2_cloudy, s2_clean, condition, sar