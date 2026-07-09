#Este archivo lo usamos para generar mascaras de nubes cuando estabamos probando adaptar lama, para el resto de modelos no se usaron.
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from skimage.filters import threshold_otsu

OTSU_OFFSET = -0.03

SPLITS = {
    "train": {
        "cloudy": Path("data/train/south_america_s2_cloudy"),
        "masks": Path("data/train/south_america_s2_masks"),
    },
    "test": {
        "cloudy": Path("data/test/south_america_s2_cloudy"),
        "masks": Path("data/test/south_america_s2_masks"),
    },
}

B03_IDX = 3

def mask_filename(cloudy_name: str) -> str:
    """ROIs1158_spring_s2_cloudy_17_p30.tif → ROIs1158_spring_s2_mask_17_p30.tif"""
    return cloudy_name.replace("_s2_cloudy_", "_s2_mask_")


def generate_mask(tif_path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(tif_path) as src:
        b03 = src.read(indexes=B03_IDX).astype(np.float32) / 10000.0
        profile = src.profile

    threshold = threshold_otsu(b03) + OTSU_OFFSET
    cloud_mask = (b03 > threshold).astype(np.uint8)  # [H, W]

    return cloud_mask, profile

def save_mask(mask: np.ndarray, profile: dict, out_path: Path) -> None:
    profile.update(
        count=1,
        dtype=rasterio.uint8,
        compress="lzw",
    )
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mask[np.newaxis, ...])  # [1, H, W]

def main() -> None:
    for split, paths in SPLITS.items():
        cloudy_folder = paths["cloudy"]
        masks_folder = paths["masks"]

        if not cloudy_folder.exists():
            continue

        masks_folder.mkdir(parents=True, exist_ok=True)

        tifs = sorted(cloudy_folder.glob("*.tif"))
        print(f"\n[{split}] {len(tifs)} imágenes en {cloudy_folder}")

        ok = 0
        skipped = 0
        errors = 0

        for i, tif_path in enumerate(tifs, 1):
            out_path = masks_folder / mask_filename(tif_path.name)

            if out_path.exists():
                skipped += 1
                continue

            try:
                mask, profile = generate_mask(tif_path)
                save_mask(mask, profile, out_path)
                ok += 1

                if i % 500 == 0 or i == len(tifs):
                    coverage = mask.mean() * 100
                    print(f"  [{i}/{len(tifs)}] {tif_path.name} → {coverage:.1f}% nuboso")

            except Exception as e:
                print(f"  ERROR en {tif_path.name}: {e}")
                errors += 1

if __name__ == "__main__":
    main()