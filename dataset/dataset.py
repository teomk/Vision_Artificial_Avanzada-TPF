from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader

S1_MEAN_DEFAULT = np.array([-8.999908447265625, -14.78221321105957], dtype=np.float32)
S1_STD_DEFAULT  = np.array([2.413282871246338, 2.3029115200042725], dtype=np.float32)

BANDS = [2, 3, 4, 8, 12, 13]
BANDS_TOTAL = [1,2,3,4,5,6,7,8,9,10,11,12,13]

def parse_filename(fname: str) -> tuple[str, str, str, str]:
    """ROIs1158_spring_s2_cloudy_17_p103.tif → (roi, season, num, patch)"""
    name  = fname.replace(".tif", "")
    parts = name.split("_")
    roi    = parts[0]
    season = parts[1]
    patch  = parts[-1]
    num    = parts[-2]
    return roi, season, num, patch


def mask_filename(cloudy_name: str) -> str:
    """ROIs1158_spring_s2_cloudy_17_p103.tif → ROIs1158_spring_s2_mask_17_p103.tif"""
    return cloudy_name.replace("_s2_cloudy_", "_s2_mask_")

def build_triple_index(s1_folder: Path, s2_folder: Path, cloudy_folder: Path, masks_folder:  Path) -> list[dict]:
    def index_folder(folder: Path) -> dict:
        idx = {}
        for f in folder.glob("*.tif"):
            try:
                key = parse_filename(f.name)
                idx[key] = f
            except Exception:
                pass
        return idx

    s1_idx     = index_folder(s1_folder)
    s2_idx     = index_folder(s2_folder)
    cloudy_idx = index_folder(cloudy_folder)

    mask_idx = {}
    for key, cloudy_path in cloudy_idx.items():
        mask_path = masks_folder / mask_filename(cloudy_path.name)
        if mask_path.exists():
            mask_idx[key] = mask_path

    valid_keys = (
        set(s1_idx.keys())
        & set(s2_idx.keys())
        & set(cloudy_idx.keys())
        & set(mask_idx.keys())
    )

    triples = [
        {
            "key":    key,
            "s1":     s1_idx[key],
            "s2":     s2_idx[key],
            "cloudy": cloudy_idx[key],
            "mask":   mask_idx[key],
        }
        for key in sorted(valid_keys)
    ]

    return triples

class SEN12MSCRDataset(Dataset):
    def __init__(
        self,
        split:      str = "train",
        include_s1: bool = True,
        s1_mean:    Optional[np.ndarray] = S1_MEAN_DEFAULT,
        s1_std:     Optional[np.ndarray] = S1_STD_DEFAULT,
        base_dir:   Path = Path("."),
        include_mask: bool = False, #Esta la opcion de recibir mascaras de nubes cuando estabamos probando adaptar lama, para el resto de modelos no se usaron.
        total_bands = False
    ):
        self.split      = split
        self.include_s1 = include_s1
        self.include_mask = include_mask
        self.s1_mean    = s1_mean
        self.s1_std     = s1_std
        self.total_bands = total_bands

        root          = base_dir / "data" / split
        s1_folder     = root / "south_america_s1"
        s2_folder     = root / "south_america_s2"
        cloudy_folder = root / "south_america_s2_cloudy"
        masks_folder  = root / "south_america_s2_masks"

        self.triples = build_triple_index(s1_folder, s2_folder, cloudy_folder, masks_folder)

        print(f"[SEN12MSCRDataset] split={split} | triples={len(self.triples)} | include_s1={include_s1} | include_mask={include_mask}")

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int):
        triple = self.triples[idx]

        bands_to_use = BANDS_TOTAL if self.total_bands else BANDS

        with rasterio.open(triple["s1"]) as src:
            s1_raw = src.read().astype(np.float32)          # [C_s1, H, W]

        with rasterio.open(triple["s2"]) as src:
            s2_raw = src.read(indexes=bands_to_use).astype(np.float32)          # [6, H, W] or [13, H, W]

        with rasterio.open(triple["cloudy"]) as src:
            cloudy_raw = src.read(indexes=bands_to_use).astype(np.float32)  # [6, H, W] or [13, H, W]

        with rasterio.open(triple["mask"]) as src:
            mask_raw = src.read().astype(np.float32)        # [1, H, W]

        s2_clear = np.clip(s2_raw    / 10000.0, 0, 1)
        s2_cloudy = np.clip(cloudy_raw / 10000.0, 0, 1)

        if self.s1_mean is not None and self.s1_std is not None:
            mean = self.s1_mean[:, None, None]
            std  = self.s1_std[:, None, None]
            s1   = (s1_raw - mean) / (std + 1e-6)
        else:
            s1 = s1_raw / 10000.0

        mask = mask_raw

        sample = {
            "s1":       torch.from_numpy(s1),
            "s2_clear": torch.from_numpy(s2_clear),
            "s2_cloudy":torch.from_numpy(s2_cloudy),
            "mask":     torch.from_numpy(mask),
            "key":      str(triple["key"]),
        }

        if self.include_s1:
            if self.include_mask:
                return (
                    sample["s1"],
                    sample["s2_cloudy"],
                    sample["mask"],
                    sample["s2_clear"],
                )
            return (
                sample["s1"],
                sample["s2_cloudy"],
                sample["s2_clear"],
            )
        else:
            if self.include_mask:
                return (
                    sample["s2_cloudy"],
                    sample["mask"],
                    sample["s2_clear"],
                )
            return (
                sample["s2_cloudy"],
                sample["s2_clear"],
             )

def compute_s1_stats(s1_folder: Path, max_samples: int = 500) -> tuple[np.ndarray, np.ndarray]:
    tifs = sorted(s1_folder.glob("*.tif"))[:max_samples]
    if not tifs:
        raise FileNotFoundError(f"No se encontraron .tif en {s1_folder}")

    all_data = []
    for tif in tifs:
        with rasterio.open(tif) as src:
            data = src.read().astype(np.float32)  # [C, H, W]
        all_data.append(data.reshape(data.shape[0], -1))  # [C, H*W]

    all_data = np.concatenate(all_data, axis=1)  # [C, N]
    mean = all_data.mean(axis=1)
    std  = all_data.std(axis=1)

    print("S1 stats calculadas sobre", len(tifs), "imágenes:")
    for i, (m, s) in enumerate(zip(mean, std)):
        print(f"  Banda {i+1}: mean={m:.2f}  std={s:.2f}")

    return mean, std