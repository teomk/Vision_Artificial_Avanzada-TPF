import rasterio
import numpy as np

path = "data/test/south_america_s2/ROIs1970_fall_s2_85_p91.tif"
BANDS = [2, 3, 4, 8, 12, 13]

# Intentar leer varias veces
for i in range(3):
    try:
        with rasterio.open(path) as src:
            data = src.read(indexes=BANDS).astype(np.float32)
        print(f"Intento {i}: OK, shape={data.shape}")
    except Exception as e:
        print(f"Intento {i}: FALLO - {e}")