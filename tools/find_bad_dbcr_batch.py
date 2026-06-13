from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset"
UTILS_DIR = ROOT / "utils"

import sys

sys.path.append(str(DATA_DIR))
sys.path.append(str(UTILS_DIR))

from dataset import BANDS, SEN12MSCRDataset


def _finite_stats(array: np.ndarray) -> dict[str, int]:
    finite = np.isfinite(array)
    return {
        "total": int(array.size),
        "nonfinite": int((~finite).sum()),
        "finite": int(finite.sum()),
    }


def _read_raw(path: Path, indexes=None) -> np.ndarray:
    with rasterio.open(path) as src:
        if indexes is None:
            return src.read().astype(np.float32)
        return src.read(indexes=indexes).astype(np.float32)


def _inspect_file(label: str, path: Path, indexes=None):
    raw = _read_raw(path, indexes=indexes)
    stats = _finite_stats(raw)
    print(f"  {label}: {path.name}")
    print(f"    shape={tuple(raw.shape)} | nonfinite={stats['nonfinite']} / {stats['total']}")
    if stats["nonfinite"] > 0:
        for band_idx in range(raw.shape[0]):
            band = raw[band_idx]
            bad = int((~np.isfinite(band)).sum())
            if bad > 0:
                print(f"    band {band_idx + 1}: nonfinite={bad}")


def _normalize_sample(s1_raw, s2_raw, cloudy_raw, mask_raw, s1_mean, s1_std):
    s2_clear = np.clip(s2_raw / 10000.0, 0, 1)
    s2_cloudy = np.clip(cloudy_raw / 10000.0, 0, 1)

    if s1_mean is not None and s1_std is not None:
        mean = s1_mean[:, None, None]
        std = s1_std[:, None, None]
        s1 = (s1_raw - mean) / (std + 1e-6)
    else:
        s1 = s1_raw / 10000.0

    mask = mask_raw.astype(np.float32)
    return s1, s2_clear, s2_cloudy, mask


def _count_nonfinite_torch(tensor: torch.Tensor | None) -> int:
    if tensor is None:
        return 0
    return int((~torch.isfinite(tensor)).sum().item())


def main():
    # run python tools/find_bad_dbcr_batch.py --config configs/dbcr_concat.yaml --batch-index 109
    parser = argparse.ArgumentParser(description="Identifica qué TIFFs forman un batch problemático y qué tensor trae NaN/Inf.")
    parser.add_argument("--config", type=str, required=True, help="Config YAML usado en la evaluación.")
    parser.add_argument("--batch-index", type=int, required=True, help="Índice de batch reportado por la evaluación (1-based).")
    parser.add_argument("--batch-size", type=int, default=None, help="Sobrescribe el batch size de config si hace falta.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    sar_mode = cfg["sar_mode"]
    batch_size = args.batch_size or cfg["train"]["batch_size"]

    ds = SEN12MSCRDataset(
        split="test",
        include_s1=(sar_mode != "None"),
        include_mask=False,
        base_dir=ROOT,
    )

    start = (args.batch_index - 1) * batch_size
    end = min(start + batch_size, len(ds))

    if start >= len(ds):
        raise SystemExit(f"Batch {args.batch_index} no existe: start={start}, len={len(ds)}")

    print(f"SAR mode: {sar_mode}")
    print(f"Dataset size: {len(ds)}")
    print(f"Batch size: {batch_size}")
    print(f"Batch {args.batch_index} -> samples [{start}, {end - 1}] ({end - start} muestras)")
    print()

    s1_mean = ds.s1_mean
    s1_std = ds.s1_std

    for sample_idx in range(start, end):
        triple = ds.triples[sample_idx]
        print(f"Sample {sample_idx} | key={triple['key']}")
        _inspect_file("s2_cloudy", triple["cloudy"], indexes=BANDS)
        _inspect_file("s2_clear ", triple["s2"], indexes=BANDS)

        if sar_mode != "None":
            _inspect_file("s1      ", triple["s1"], indexes=None)

        s1_raw = _read_raw(triple["s1"])
        s2_raw = _read_raw(triple["s2"], indexes=BANDS)
        cloudy_raw = _read_raw(triple["cloudy"], indexes=BANDS)
        mask_raw = _read_raw(triple["mask"])

        s1, s2_clear, s2_cloudy, mask = _normalize_sample(s1_raw, s2_raw, cloudy_raw, mask_raw, s1_mean, s1_std)

        tensors = {
            "s1": torch.from_numpy(s1),
            "s2_clear": torch.from_numpy(s2_clear),
            "s2_cloudy": torch.from_numpy(s2_cloudy),
            "mask": torch.from_numpy(mask),
        }

        if sar_mode == "None":
            condition = tensors["s2_cloudy"]
            sar = None
        elif sar_mode == "Concat":
            condition = torch.cat([tensors["s2_cloudy"], tensors["s1"]], dim=0)
            sar = None
        else:
            condition = tensors["s2_cloudy"]
            sar = tensors["s1"]

        counts = {
            "s1": _count_nonfinite_torch(tensors["s1"]),
            "s2_clear": _count_nonfinite_torch(tensors["s2_clear"]),
            "s2_cloudy": _count_nonfinite_torch(tensors["s2_cloudy"]),
            "mask": _count_nonfinite_torch(tensors["mask"]),
            "condition": _count_nonfinite_torch(condition),
            "sar": _count_nonfinite_torch(sar),
        }

        bad = {name: count for name, count in counts.items() if count > 0}
        if bad:
            print(f"  Nonfinite en tensores: {bad}")
        else:
            print("  Todos los tensores derivados son finitos.")

        print()


if __name__ == "__main__":
    main()