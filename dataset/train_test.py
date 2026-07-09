from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import shutil

SEASONS = {"spring", "summer", "fall", "autumn", "winter"}

S1_TEST = Path("data/test/south_america_s1")
S2_TEST = Path("data/test/south_america_s2")
CLOUDY_TEST = Path("data/test/south_america_s2_cloudy")

S1_TRAIN = Path("data/train/south_america_s1")
S2_TRAIN = Path("data/train/south_america_s2")
CLOUDY_TRAIN = Path("data/train/south_america_s2_cloudy")

TEST_ACQUISITIONS = {
    "spring": "44",
    "summer": "146",
    "fall": "85",
    "winter": "64",
}
def parse_filename(fname: str) -> tuple[str, str, str, str]:
    name = fname.replace(".tif", "")
    parts = name.split("_")
    roi = parts[0]
    season = parts[1]
    patch = parts[-1]
    num = parts[-2]
    return roi, season, num, patch


def build_index(folder: Path) -> dict:
    index = {}
    for f in folder.glob("*.tif"):
        try:
            roi, season, num, patch = parse_filename(f.name)
            key = (roi, season, num, patch)
            index[key] = f
        except Exception:
            pass
    return index


def main() -> None:
    s1_idx = build_index(S1_TEST)
    s2_idx = build_index(S2_TEST)
    cloudy_idx = build_index(CLOUDY_TEST)

    triples = set(s1_idx) & set(s2_idx) & set(cloudy_idx)
    print(f"Triples en test actualmente: {len(triples)}")

    # Crear carpetas train
    for folder in [S1_TRAIN, S2_TRAIN, CLOUDY_TRAIN]:
        folder.mkdir(parents=True, exist_ok=True)

    to_train = set()
    to_keep = set()
    for key in triples:
        roi, season, num, patch = key
        if TEST_ACQUISITIONS.get(season) == num:
            to_keep.add(key)
        else:
            to_train.add(key)

    print(f"Triples que van a train: {len(to_train)}")
    print(f"Triples que se quedan en test: {len(to_keep)}")

    for key in to_train:
        roi, season, num, patch = key

        src = s1_idx[key]
        shutil.move(str(src), S1_TRAIN / src.name)

        src = s2_idx[key]
        shutil.move(str(src), S2_TRAIN / src.name)

        src = cloudy_idx[key]
        shutil.move(str(src), CLOUDY_TRAIN / src.name)

    print("\nListo. Archivos movidos a data/train/")

if __name__ == "__main__":
    main()