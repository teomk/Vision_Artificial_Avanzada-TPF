from pathlib import Path
import rasterio
from tqdm import tqdm

root = Path("data")
tifs = list(root.rglob("*.tif"))

bad = []

for path in tqdm(tifs):
    try:
        with rasterio.open(path) as src:
            _ = src.read()
    except Exception as e:
        bad.append((str(path), str(e)))

print(f"\nArchivos corruptos: {len(bad)}")

for p, err in bad:
    print("\n---")
    print(p)
    print(err)