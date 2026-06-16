from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader

# ─────────────────────────────────────────
# NORMALIZACIÓN S1
# Opción A (recomendada): media/std globales precalculadas del dataset
# Opción B (rápida):      dividir por 10000 igual que S2
#
# Para usar Opción A, calculá primero las estadísticas con compute_s1_stats()
# y pasalas como s1_mean y s1_std al Dataset.
# Para usar Opción B, dejá s1_mean=None y s1_std=None.
# ─────────────────────────────────────────

# Estadísticas S1 por defecto (None = usar /10000)
S1_MEAN_DEFAULT = np.array([-8.999908447265625, -14.78221321105957], dtype=np.float32)
S1_STD_DEFAULT  = np.array([2.413282871246338, 2.3029115200042725], dtype=np.float32)
BANDS = [2, 3, 4, 8, 12, 13]

BANDS_TOTAL = [1,2,3,4,5,6,7,8,9,10,11,12,13]


# ─────────────────────────────────────────
# HELPERS DE PARSING
# ─────────────────────────────────────────

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


# ─────────────────────────────────────────
# CONSTRUCCIÓN DEL ÍNDICE DE TRIPLES
# ─────────────────────────────────────────

def build_triple_index(
    s1_folder:     Path,
    s2_folder:     Path,
    cloudy_folder: Path,
    masks_folder:  Path,
) -> list[dict]:
    """
    Recorre las carpetas y devuelve una lista de dicts, uno por triplete completo:
        {
            "key":    (roi, season, num, patch),
            "s1":     Path,
            "s2":     Path,
            "cloudy": Path,
            "mask":   Path,
        }
    Solo incluye tripletes donde los 4 archivos existen.
    """
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

    # Índice de máscaras: mismo key que cloudy
    mask_idx = {}
    for key, cloudy_path in cloudy_idx.items():
        mask_path = masks_folder / mask_filename(cloudy_path.name)
        if mask_path.exists():
            mask_idx[key] = mask_path

    # Intersección de los 4
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
        for key in sorted(valid_keys)  # sorted para reproducibilidad
    ]

    return triples


# ─────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────

class SEN12MSCRDataset(Dataset):
    """
    Dataset para SEN12MS-CR (cloud removal).

    Parámetros
    ----------
    split : "train" | "test"
        Subcarpeta base dentro de data/.
    include_s1 : bool
        Si True  → devuelve (s1, s2_cloudy, mask, s2_clear)   [con SAR]
        Si False → devuelve (s2_cloudy, mask, s2_clear)        [solo óptico]
    transform : callable, opcional
        Función que recibe un dict con todas las tensores y devuelve el dict
        transformado. Útil para data augmentation futura.
        Ejemplo de firma:
            def my_transform(sample: dict) -> dict: ...
        Las claves disponibles son: "s1", "s2_clear", "s2_cloudy", "mask"
    s1_mean : np.ndarray | None
        Media por banda para normalizar S1. Shape [C_s1].
        Si None, se normaliza dividiendo por 10000.
    s1_std : np.ndarray | None
        Std por banda para normalizar S1. Shape [C_s1].
        Si None, se normaliza dividiendo por 10000.
    base_dir : Path
        Directorio raíz del proyecto. Por defecto el directorio actual.
    """

    def __init__(
        self,
        split:      str = "train",
        include_s1: bool = True,
        transform:  Optional[Callable] = None,
        s1_mean:    Optional[np.ndarray] = S1_MEAN_DEFAULT,
        s1_std:     Optional[np.ndarray] = S1_STD_DEFAULT,
        base_dir:   Path = Path("."),
        include_mask: bool = True,
        total_bands = False
    ):
        self.split      = split
        self.include_s1 = include_s1
        self.include_mask = include_mask
        self.transform  = transform
        self.s1_mean    = s1_mean
        self.s1_std     = s1_std
        self.total_bands = total_bands

        root          = base_dir / "data" / split
        s1_folder     = root / "south_america_s1"
        s2_folder     = root / "south_america_s2"
        cloudy_folder = root / "south_america_s2_cloudy"
        masks_folder  = root / "south_america_s2_masks"

        self.triples = build_triple_index(
            s1_folder, s2_folder, cloudy_folder, masks_folder
        )

        if len(self.triples) == 0:
            raise RuntimeError(
                f"No se encontraron tripletes completos en '{root}'. "
                "Verificá que las carpetas y máscaras estén generadas."
            )

        print(f"[SEN12MSCRDataset] split={split} | triples={len(self.triples)} | include_s1={include_s1} | include_mask={include_mask}")

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int):
        triple = self.triples[idx]

        bands_to_use = BANDS_TOTAL if self.total_bands else BANDS

        # ── Leer imágenes ──────────────────────────────────────────────
        with rasterio.open(triple["s1"]) as src:
            s1_raw = src.read().astype(np.float32)          # [C_s1, H, W]

        with rasterio.open(triple["s2"]) as src:
            s2_raw = src.read(indexes=bands_to_use).astype(np.float32)          # [6, H, W] or [13, H, W]

        with rasterio.open(triple["cloudy"]) as src:
            cloudy_raw = src.read(indexes=bands_to_use).astype(np.float32)  # [6, H, W] or [13, H, W]

        with rasterio.open(triple["mask"]) as src:
            mask_raw = src.read().astype(np.float32)        # [1, H, W]

        # ── Normalización ──────────────────────────────────────────────
        # S2: reflectancia física [0, 1]
        s2_clear = np.clip(s2_raw    / 10000.0, 0, 1)
        s2_cloudy = np.clip(cloudy_raw / 10000.0, 0, 1)

        # S1: media/std si están disponibles, sino /10000
        if self.s1_mean is not None and self.s1_std is not None:
            mean = self.s1_mean[:, None, None]  # broadcast [C, 1, 1]
            std  = self.s1_std[:, None, None]
            s1   = (s1_raw - mean) / (std + 1e-6)
        else:
            s1 = s1_raw / 10000.0

        # Máscara: ya es binaria {0, 1}, solo aseguramos float
        mask = mask_raw  # [1, H, W]

        # ── Convertir a tensores ───────────────────────────────────────
        sample = {
            "s1":       torch.from_numpy(s1),
            "s2_clear": torch.from_numpy(s2_clear),
            "s2_cloudy":torch.from_numpy(s2_cloudy),
            "mask":     torch.from_numpy(mask),
            "key":      str(triple["key"]),  # útil para debug
        }

        # ── Augmentation (hook para el futuro) ────────────────────────
        if self.transform is not None:
            sample = self.transform(sample)

        # ── Output según flag ──────────────────────────────────────────
        if self.include_s1:
            if self.include_mask:
                # Con SAR y máscara: (s1, s2_cloudy, mask) → target: s2_clear
                return (
                    sample["s1"],
                    sample["s2_cloudy"],
                    sample["mask"],
                    sample["s2_clear"],
                )
            # Con SAR: (s1, s2_cloudy) → target: s2_clear
            return (
                sample["s1"],
                sample["s2_cloudy"],
                sample["s2_clear"],
            )
        else:
            if self.include_mask:
                # Sin SAR pero con máscara: (s2_cloudy, mask) → target: s2_clear
                return (
                    sample["s2_cloudy"],
                    sample["mask"],
                    sample["s2_clear"],
                )
            # Sin SAR: (s2_cloudy) → target: s2_clear
            return (
                sample["s2_cloudy"],
                sample["s2_clear"],
             )


# ─────────────────────────────────────────
# UTILIDAD: calcular estadísticas S1
# ─────────────────────────────────────────

def compute_s1_stats(s1_folder: Path, max_samples: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula media y std por banda sobre una muestra del dataset S1.
    Usá esto una sola vez y hardcodeá los valores en S1_MEAN_DEFAULT / S1_STD_DEFAULT.

    Ejemplo de uso:
        mean, std = compute_s1_stats(Path("data/train/south_america_s1"))
        print("S1_MEAN =", mean.tolist())
        print("S1_STD  =", std.tolist())
    """
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


# ─────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────

if __name__ == "__main__":

    mean, std = compute_s1_stats(Path("data/train/south_america_s1"))
    print("S1_MEAN =", mean.tolist())
    print("S1_STD  =", std.tolist())

    # --- Con S1 ---
    ds_with_s1 = SEN12MSCRDataset(split="test", include_s1=True, bands_total=True)
    s1, cloudy, mask, clear = ds_with_s1[0]
    print(f"\nCon S1:")
    print(f"  s1:       {s1.shape}    dtype={s1.dtype}    min={s1.min():.3f}  max={s1.max():.3f}")
    print(f"  cloudy:   {cloudy.shape} dtype={cloudy.dtype} min={cloudy.min():.3f}  max={cloudy.max():.3f}")
    print(f"  mask:     {mask.shape}   dtype={mask.dtype}   unique={mask.unique().tolist()}")
    print(f"  s2_clear: {clear.shape}  dtype={clear.dtype}  min={clear.min():.3f}  max={clear.max():.3f}")

    # --- Sin S1 ---
    ds_no_s1 = SEN12MSCRDataset(split="test", include_s1=False, bands_total=True)
    cloudy, mask, clear = ds_no_s1[0]
    print(f"\nSin S1:")
    print(f"  cloudy:   {cloudy.shape}")
    print(f"  mask:     {mask.shape}")
    print(f"  s2_clear: {clear.shape}")

    # --- DataLoader ---
    loader = DataLoader(ds_with_s1, batch_size=4, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    s1_b, cloudy_b, mask_b, clear_b = batch
    print(f"\nBatch con S1:")
    print(f"  s1:       {s1_b.shape}")
    print(f"  cloudy:   {cloudy_b.shape}")
    print(f"  mask:     {mask_b.shape}")
    print(f"  s2_clear: {clear_b.shape}")