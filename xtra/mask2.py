import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.filters import threshold_otsu

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────

TIFF_PATH = "data/train/south_america_s2_cloudy/ROIs1158_spring_s2_cloudy_17_p106.tif"
# TIFF_PATH = "data/train/south_america_s2_cloudy/ROIs1158_spring_s2_cloudy_17_p136.tif"
# TIFF_PATH = "data/train/south_america_s2_cloudy/ROIs1158_spring_s2_cloudy_17_p166.tif"
# TIFF_PATH = "data/train/south_america_s2_cloudy/ROIs2017_winter_s2_cloudy_49_p200.tif"
# TIFF_PATH = "data/train/south_america_s2_cloudy/ROIs2017_winter_s2_cloudy_49_p150.tif"
# TIFF_PATH = "data/train/south_america_s2_cloudy/ROIs2017_winter_s2_cloudy_49_p500.tif"

RGB_IDX = [4, 3, 2]  # B04=Red, B03=Green, B02=Blue
B03_IDX = 3           # banda verde, sensible a nubes


# ─────────────────────────────────────────
# PASO 1: leer imagen
# ─────────────────────────────────────────

print("[1/3] Leyendo imagen...")

with rasterio.open(TIFF_PATH) as src:
    rgb_data = src.read(indexes=RGB_IDX).astype(np.float32) / 10000.0
    b03      = src.read(indexes=B03_IDX).astype(np.float32) / 10000.0

print(f"  B03 min={b03.min():.4f}  max={b03.max():.4f}  mean={b03.mean():.4f}")


# ─────────────────────────────────────────
# PASO 2: generar máscara con Otsu
# ─────────────────────────────────────────

print("[2/3] Calculando threshold con Otsu...")

threshold = threshold_otsu(b03) - 0.03
print(f"  Threshold Otsu: {threshold:.4f}")

cloud_mask = (b03 > threshold).astype(np.float32)
coverage   = cloud_mask.mean() * 100
print(f"  Cobertura de nubes: {coverage:.1f}%")


# ─────────────────────────────────────────
# PASO 3: plotear
# ─────────────────────────────────────────

print("[3/3] Ploteando...")

def stretch(img, p_low=2, p_high=98):
    out = np.zeros_like(img)
    for i in range(img.shape[0]):
        lo, hi = np.percentile(img[i], [p_low, p_high])
        out[i] = np.clip((img[i] - lo) / (hi - lo + 1e-6), 0, 1)
    return out

rgb_stretched = stretch(rgb_data).transpose(1, 2, 0)

rgb_with_mask = rgb_stretched.copy()
rgb_with_mask[cloud_mask == 1, 0] = 1.0
rgb_with_mask[cloud_mask == 1, 1] *= 0.3
rgb_with_mask[cloud_mask == 1, 2] *= 0.3

fig = plt.figure(figsize=(20, 5))
gs  = gridspec.GridSpec(1, 5, figure=fig, wspace=0.05)

ax1 = fig.add_subplot(gs[0])
ax1.imshow(rgb_stretched)
ax1.set_title("S2 Nuboso (RGB)", fontsize=12, fontweight="bold")
ax1.axis("off")

ax2 = fig.add_subplot(gs[1])
ax2.hist(b03.flatten(), bins=100, color="steelblue", edgecolor="none")
ax2.axvline(threshold, color="red", linewidth=2, label=f"Otsu={threshold:.3f}")
ax2.set_title("Histograma B03 + Otsu", fontsize=12, fontweight="bold")
ax2.set_xlabel("Reflectancia")
ax2.legend()

ax3 = fig.add_subplot(gs[2])
ax3.imshow(b03, cmap="gray")
ax3.set_title("B03 (verde)", fontsize=12, fontweight="bold")
ax3.axis("off")

ax4 = fig.add_subplot(gs[3])
ax4.imshow(cloud_mask, cmap="gray", vmin=0, vmax=1)
ax4.set_title(f"Máscara Binaria\n(Otsu={threshold:.3f})", fontsize=12, fontweight="bold")
ax4.axis("off")

ax5 = fig.add_subplot(gs[4])
ax5.imshow(rgb_with_mask)
ax5.set_title(f"RGB + Máscara\n({coverage:.1f}% nuboso)", fontsize=12, fontweight="bold")
ax5.axis("off")

plt.suptitle(
    f"Detección de nubes (Otsu): {TIFF_PATH.split('/')[-1]}",
    fontsize=13, y=1.02
)

plt.show()

print(f"\n✅ Listo!")
print(f"   Threshold Otsu: {threshold:.4f}")
print(f"   Cobertura: {coverage:.1f}%")