import rasterio

path = "data/test/south_america_s2/ROIs1970_fall_s2_85_p91.tif"  # ajustá la ruta real
try:
    with rasterio.open(path) as src:
        data = src.read()
        print(f"OK: shape={data.shape}")
except Exception as e:
    print(f"CORRUPTO: {e}")

# Verificar tamaño del archivo
from pathlib import Path
print(f"Tamaño: {Path(path).stat().st_size} bytes")