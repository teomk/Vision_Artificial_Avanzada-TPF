from __future__ import annotations

import argparse
import re
from pathlib import Path


_PATCH_SUFFIX = re.compile(r"_p\d+$")


def roi_id_from_name(path: Path) -> str:
    """Return the ROI identifier for a patch file name.

    Example:
        ROIs2017_winter_s1_21_p100.tif -> ROIs2017_winter_s1_21
    """
    stem = path.stem
    return _PATCH_SUFFIX.sub("", stem)


def count_rois(folder: Path) -> int:
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    roi_ids = {
        roi_id_from_name(path)
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() == ".tif"
    }
    return len(roi_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Count ROI .tif files in an extracted dataset folder.")
    parser.add_argument(
        "folder",
        nargs="?",
        default="data/south_america_s1",
        help="Path to one extracted output folder, e.g. data/south_america_s1",
    )
    args = parser.parse_args()

    folder = Path(args.folder)
    total = count_rois(folder)
    print(f"{folder}: {total} distinct ROIs")


if __name__ == "__main__":
    main()