from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from dataset import BANDS, BANDS_TOTAL, S1_MEAN_DEFAULT, S1_STD_DEFAULT, build_triple_index


class PathAwareSEN12MSCRDataset(Dataset):
    """SEN12MS-CR dataset that keeps file paths for each sample."""

    def __init__(
        self,
        split: str = "train",
        transform: Optional[Callable] = None,
        s1_mean: Optional[np.ndarray] = S1_MEAN_DEFAULT,
        s1_std: Optional[np.ndarray] = S1_STD_DEFAULT,
        base_dir: Path = Path("."),
        total_bands: bool = False,
    ):
        self.split = split
        self.transform = transform
        self.s1_mean = s1_mean
        self.s1_std = s1_std
        self.total_bands = total_bands

        root = base_dir / "data" / split
        s1_folder = root / "south_america_s1"
        s2_folder = root / "south_america_s2"
        cloudy_folder = root / "south_america_s2_cloudy"
        masks_folder = root / "south_america_s2_masks"

        self.triples = build_triple_index(s1_folder, s2_folder, cloudy_folder, masks_folder)

        if len(self.triples) == 0:
            raise RuntimeError(
                f"No se encontraron tripletes completos en '{root}'. Verificá que las carpetas y máscaras estén generadas."
            )

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int):
        triple = self.triples[idx]
        bands_to_use = BANDS_TOTAL if self.total_bands else BANDS

        with rasterio.open(triple["s1"]) as src:
            s1_raw = src.read().astype(np.float32)

        with rasterio.open(triple["s2"]) as src:
            s2_raw = src.read(indexes=bands_to_use).astype(np.float32)

        with rasterio.open(triple["cloudy"]) as src:
            cloudy_raw = src.read(indexes=bands_to_use).astype(np.float32)

        with rasterio.open(triple["mask"]) as src:
            mask_raw = src.read().astype(np.float32)

        s2_clear = np.clip(s2_raw / 10000.0, 0, 1)
        s2_cloudy = np.clip(cloudy_raw / 10000.0, 0, 1)

        if self.s1_mean is not None and self.s1_std is not None:
            mean = self.s1_mean[:, None, None]
            std = self.s1_std[:, None, None]
            s1 = (s1_raw - mean) / (std + 1e-6)
        else:
            s1 = s1_raw / 10000.0

        sample = {
            "s1": torch.from_numpy(s1),
            "s2_clear": torch.from_numpy(s2_clear),
            "s2_cloudy": torch.from_numpy(s2_cloudy),
            "mask": torch.from_numpy(mask_raw),
            "key": str(triple["key"]),
            "paths": {
                "s1": str(triple["s1"]),
                "s2": str(triple["s2"]),
                "s2_cloudy": str(triple["cloudy"]),
                "mask": str(triple["mask"]),
            },
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample