import os
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import rasterio
except Exception:
    rasterio = None


def _read_raster(path: Path) -> np.ndarray:
    if rasterio is None:
        raise RuntimeError("rasterio is required to read geotiffs. Install from requirements.txt")
    with rasterio.open(path) as src:
        arr = src.read()
    # rasterio returns (C, H, W)
    return arr.astype(np.float32)


class SatellitePatchDataset(Dataset):
    """Dataset that returns aligned patches: clean S2 (target), cloudy S2, cloud mask, optional SAR.

    Expects three folders under `root`:
      - south_america_s2: clean target images (3 bands)
      - south_america_s2_cloudy: cloudy S2 (3 bands)
      - south_america_s1: SAR (2 bands) (optional)

    Filenames must match across folders (same stem). Cloud mask is expected as a single-band file
    with the same stem and either in the cloudy folder (as a 4th band) or as a separate file
    with suffix `_mask` (e.g. stem_mask.tif).
    """

    def __init__(self, root: str, split: str = "train", transform=None, patch_size: Optional[int] = None):
        self.root = Path(root)
        self.s2_dir = self.root / "south_america_s2"
        self.s2_cloudy_dir = self.root / "south_america_s2_cloudy"
        self.s1_dir = self.root / "south_america_s1"

        self.transform = transform
        self.patch_size = patch_size

        # Build index from cloudy folder
        self.paths = [p for p in sorted(self.s2_cloudy_dir.iterdir()) if p.suffix.lower() == ".tif"]

    def __len__(self) -> int:
        return len(self.paths)

    def _load_mask(self, stem: str, cloudy_path: Path) -> np.ndarray:
        # Try to find mask as separate file stem_mask.tif
        mask_path = cloudy_path.with_name(stem + "_mask.tif")
        if mask_path.exists():
            m = _read_raster(mask_path)
            if m.ndim == 3:
                m = m[0:1]
            return m

        # If cloudy has 4 bands, assume last is mask
        arr = _read_raster(cloudy_path)
        if arr.shape[0] == 4:
            return arr[3:4]

        # Otherwise create a mask of zeros (no cloud) as fallback
        c, h, w = arr.shape
        return np.zeros((1, h, w), dtype=np.float32)

    def __getitem__(self, idx: int):
        cloudy_path = self.paths[idx]
        stem = cloudy_path.stem

        s2_cloudy = _read_raster(cloudy_path)
        s2_clean_path = self.s2_dir / cloudy_path.name
        if not s2_clean_path.exists():
            raise FileNotFoundError(f"Clean S2 not found for {cloudy_path.name}")
        s2_clean = _read_raster(s2_clean_path)

        mask = self._load_mask(stem, cloudy_path)

        sar_path = self.s1_dir / cloudy_path.name
        sar = None
        if sar_path.exists():
            sar = _read_raster(sar_path)

        # Optionally crop patches
        if self.patch_size is not None:
            _, H, W = s2_clean.shape
            ph = min(self.patch_size, H)
            pw = min(self.patch_size, W)
            top = np.random.randint(0, H - ph + 1)
            left = np.random.randint(0, W - pw + 1)

            def crop(a):
                return a[:, top:top + ph, left:left + pw]

            s2_clean = crop(s2_clean)
            s2_cloudy = crop(s2_cloudy)
            mask = crop(mask)
            if sar is not None:
                sar = crop(sar)

        # Random horizontal flip
        if np.random.rand() > 0.5:
            s2_clean = s2_clean[:, :, ::-1]
            s2_cloudy = s2_cloudy[:, :, ::-1]
            mask = mask[:, :, ::-1]
            if sar is not None:
                sar = sar[:, :, ::-1]

        # Convert to torch and normalize: images -> [-1,1], mask -> [0,1]
        def to_tensor(a):
            return torch.from_numpy(a.copy())

        x0 = to_tensor(s2_clean) / 127.5 - 1.0
        s2_cloudy_t = to_tensor(s2_cloudy) / 127.5 - 1.0
        mask_t = to_tensor(mask) / 255.0
        if sar is not None:
            sar_t = to_tensor(sar) / 32768.0  # SAR scale heuristic; adjust if needed
        else:
            sar_t = None

        return {
            "x0": x0,              # clean target [C,H,W] float32
            "s2_cloudy": s2_cloudy_t,
            "mask": mask_t,
            "sar": sar_t,
            "filename": cloudy_path.name,
        }
