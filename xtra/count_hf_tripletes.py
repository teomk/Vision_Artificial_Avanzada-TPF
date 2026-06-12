from huggingface_hub import HfApi
from collections import defaultdict

REPO_ID = "LucioLuque/sen12mscr-south-america"
REPO_TYPE = "dataset"


def parse_filename(path):
    """
    Ejemplos:
    train/south_america_s1/ROIs1158_spring_s1_17_p30.tif
    train/south_america_s2/ROIs1158_spring_s2_17_p30.tif
    train/south_america_s2_cloudy/ROIs1158_spring_s2_cloudy_17_p30.tif
    train/south_america_s2_masks/ROIs1158_spring_s2_mask_17_p30.tif

    Devuelve una clave común:
    (roi, season, num, patch)
    """
    fname = path.replace("\\", "/").split("/")[-1]

    if not fname.endswith(".tif"):
        return None

    name = fname.replace(".tif", "")
    parts = name.split("_")

    try:
        roi = parts[0]
        season = parts[1]
        patch = parts[-1]
        num = parts[-2]
    except IndexError:
        return None

    return roi, season, num, patch


def count_split(files, split):
    """
    split puede ser:
    - train
    - test
    - validation, si más adelante existe esa carpeta
    """

    folders = {
        "s1": f"{split}/south_america_s1/",
        "s2": f"{split}/south_america_s2/",
        "cloudy": f"{split}/south_america_s2_cloudy/",
        "mask": f"{split}/south_america_s2_masks/",
    }

    indexes = {
        "s1": {},
        "s2": {},
        "cloudy": {},
        "mask": {},
    }

    for path in files:
        path = path.replace("\\", "/")

        for kind, prefix in folders.items():
            if path.startswith(prefix) and path.endswith(".tif"):
                key = parse_filename(path)
                if key is not None:
                    indexes[kind][key] = path

    s1_keys = set(indexes["s1"].keys())
    s2_keys = set(indexes["s2"].keys())
    cloudy_keys = set(indexes["cloudy"].keys())
    mask_keys = set(indexes["mask"].keys())

    triplets_sar = s1_keys & s2_keys & cloudy_keys
    triplets_sar_mask = s1_keys & s2_keys & cloudy_keys & mask_keys
    pairs_no_sar = s2_keys & cloudy_keys

    all_keys = s1_keys | s2_keys | cloudy_keys | mask_keys
    incomplete_sar = all_keys - triplets_sar
    incomplete_sar_mask = all_keys - triplets_sar_mask

    print("=" * 80)
    print(f"SPLIT: {split}")
    print("=" * 80)

    print(f"S1:                         {len(s1_keys)}")
    print(f"S2 clean:                   {len(s2_keys)}")
    print(f"S2 cloudy:                  {len(cloudy_keys)}")
    print(f"Mask:                       {len(mask_keys)}")
    print()
    print(f"Pares S2 + cloudy:          {len(pairs_no_sar)}")
    print(f"Tripletes S1 + S2 + cloudy: {len(triplets_sar)}")
    print(f"Cuádruples + mask:          {len(triplets_sar_mask)}")
    print()
    print(f"Incompletos sin exigir mask:{len(incomplete_sar)}")
    print(f"Incompletos exigiendo mask: {len(incomplete_sar_mask)}")

    # Desglose útil
    solo_s1 = s1_keys - s2_keys - cloudy_keys
    solo_s2 = s2_keys - s1_keys - cloudy_keys
    solo_cloudy = cloudy_keys - s1_keys - s2_keys

    s1_s2_sin_cloudy = (s1_keys & s2_keys) - cloudy_keys
    s1_cloudy_sin_s2 = (s1_keys & cloudy_keys) - s2_keys
    s2_cloudy_sin_s1 = (s2_keys & cloudy_keys) - s1_keys

    print()
    print("Desglose sin exigir mask:")
    print(f"Solo S1:                    {len(solo_s1)}")
    print(f"Solo S2:                    {len(solo_s2)}")
    print(f"Solo cloudy:                {len(solo_cloudy)}")
    print(f"S1 + S2 sin cloudy:         {len(s1_s2_sin_cloudy)}")
    print(f"S1 + cloudy sin S2:         {len(s1_cloudy_sin_s2)}")
    print(f"S2 + cloudy sin S1:         {len(s2_cloudy_sin_s1)}")

    return {
        "s1": len(s1_keys),
        "s2": len(s2_keys),
        "cloudy": len(cloudy_keys),
        "mask": len(mask_keys),
        "pairs_no_sar": len(pairs_no_sar),
        "triplets_sar": len(triplets_sar),
        "triplets_sar_mask": len(triplets_sar_mask),
        "incomplete_sar": len(incomplete_sar),
        "incomplete_sar_mask": len(incomplete_sar_mask),
    }


def main():
    api = HfApi()

    print("Listando archivos remotos de Hugging Face...")
    files = api.list_repo_files(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
    )

    tif_files = [f for f in files if f.endswith(".tif")]

    print(f"Archivos totales en repo: {len(files)}")
    print(f"Archivos .tif en repo:    {len(tif_files)}")
    print()

    # Cambiá o agregá splits según tu estructura real.
    for split in ["train", "validation", "test"]:
        has_split = any(f.startswith(f"{split}/") for f in tif_files)
        if has_split:
            count_split(tif_files, split)
        else:
            print(f"No encontré split remoto: {split}")


if __name__ == "__main__":
    main()