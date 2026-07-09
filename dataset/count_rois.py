from __future__ import annotations
from collections import defaultdict
from pathlib import Path

SEASONS = {"spring", "summer", "fall", "autumn", "winter"}
ROOT = Path(__file__).resolve().parent.parent

def get_folder_paths(split: str) -> tuple[dict, dict, dict]:
    s1_path = ROOT / "data" / split / "south_america_s1"
    s2_path = ROOT / "data" / split / "south_america_s2"
    cloudy_path = ROOT / "data" / split / "south_america_s2_cloudy"
    s1_idx = build_index(s1_path)
    s2_idx = build_index(s2_path)       
    cloudy_idx = build_index(cloudy_path)
    return s1_idx, s2_idx, cloudy_idx

def parse_filename(fname: str) -> tuple[str, str, str, str]:
    name = Path(fname).stem
    parts = name.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected filename format: {fname}")
    roi = parts[0]
    season = parts[1]
    num = parts[-2]
    patch = parts[-1]
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
    total_imgs = 0
    total_triplets_all = 0
    for split in ["train", "test"]:
        s1_idx, s2_idx, cloudy_idx = get_folder_paths(split)

        triples = set(s1_idx) & set(s2_idx) & set(cloudy_idx)
        print(f"=== {split.upper()} === Tripletes {len(triples)} === Imágenes {len(triples) * 3} ==============================")

        by_season_num: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for key in triples:
            roi, season, num, patch = key
            by_season_num[season][num].append(key)

        for season in ["spring", "summer", "fall", "winter"]:
            nums = by_season_num.get(season, {})
            total_triplets = sum(len(keys) for keys in nums.values())
            total_imgs += total_triplets * 3
            total_triplets_all += total_triplets 

            print(f"{season.upper()} — {len(nums)} ROIs — {total_triplets} tripletes — {total_triplets * 3} imágenes")
            roi_num = 0
            for num in sorted(nums.keys(), key=lambda x: int(x)):

                triples_in_num = nums[num]
                roi_num += 1

                print(f"ROI {roi_num}: {len(triples_in_num)} tripletes — {len(triples_in_num) * 3} imágenes")
            print()
    print(f"Total tripletes: {total_triplets_all} — Total imágenes: {total_imgs}")

if __name__ == "__main__":
    main()