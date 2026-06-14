from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from tqdm import tqdm


def iter_tif_files(root: Path) -> list[Path]:
    return sorted(list(root.rglob("*.tif")) + list(root.rglob("*.tiff")))


def check_tif(path: Path) -> tuple[str, dict[str, object] | None]:
    try:
        with rasterio.open(path) as src:
            invalid_bands: list[dict[str, int]] = []
            total_invalid = 0

            for band_idx in range(1, src.count + 1):
                band = src.read(band_idx, masked=False)
                mask = ~np.isfinite(band)
                invalid_count = int(mask.sum())
                if invalid_count > 0:
                    invalid_bands.append({"band": band_idx, "invalid_values": invalid_count})
                    total_invalid += invalid_count

            if invalid_bands:
                return "invalid", {
                    "path": str(path),
                    "bands": invalid_bands,
                    "total_invalid_values": total_invalid,
                    "shape": (src.count, src.height, src.width),
                }

            return "ok", None

    except Exception as exc:
        return "corrupt", {"path": str(path), "error": str(exc)}


def main() -> int:
    # run python tools/check_tif_integrity.py
    parser = argparse.ArgumentParser(description="Check TIFF files for corruption, NaN, and infinity values.")
    parser.add_argument("--root", type=str, default="data", help="Root folder to scan (default: data)")
    parser.add_argument("--extensions", type=str, nargs="*", default=[".tif", ".tiff"], help="File extensions to scan")
    args = parser.parse_args()

    root = Path(args.root)
    extensions = {ext.lower() for ext in args.extensions}
    tif_files = [p for p in iter_tif_files(root) if p.suffix.lower() in extensions]

    corrupt_files: list[dict[str, object]] = []
    invalid_files: list[dict[str, object]] = []

    for path in tqdm(tif_files, desc="Checking TIFFs", unit="file"):
        status, payload = check_tif(path)
        if status == "corrupt" and payload is not None:
            corrupt_files.append(payload)
        elif status == "invalid" and payload is not None:
            invalid_files.append(payload)

    print(f"\nArchivos analizados: {len(tif_files)}")
    print(f"Archivos corruptos: {len(corrupt_files)}")
    print(f"Archivos con NaN/inf: {len(invalid_files)}")

    if corrupt_files:
        print("\n=== CORRUPTOS ===")
        for item in corrupt_files:
            print("\n---")
            print(item["path"])
            print(item["error"])

    if invalid_files:
        print("\n=== NaN / INF ===")
        for item in invalid_files:
            print("\n---")
            print(item["path"])
            print(f"shape: {item['shape']}")
            print(f"total_invalid_values: {item['total_invalid_values']}")
            for band_info in item["bands"]:
                print(f"band {band_info['band']}: {band_info['invalid_values']} invalid values")

    return 1 if (corrupt_files or invalid_files) else 0


if __name__ == "__main__":
    raise SystemExit(main())
