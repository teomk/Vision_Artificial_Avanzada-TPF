import torch
import rasterio
import numpy as np
from pathlib import Path

import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LAMA_DIR = ROOT / "external" / "lama"

sys.path.append(str(LAMA_DIR))


from saicinpainting.training.modules.ffc import FFCResNetGenerator


model_s2 = FFCResNetGenerator(
    input_nc=7,
    output_nc=6,
    ngf=64,
    n_downsampling=3,
    n_blocks=18,

    init_conv_kwargs={
        "ratio_gin": 0,
        "ratio_gout": 0,
    },

    downsample_conv_kwargs={
        "ratio_gin": 0,
        "ratio_gout": 0,
    },

    resnet_conv_kwargs={
        "ratio_gin": 0.75,
        "ratio_gout": 0.75,
        "enable_lfu": False,
    }
)

TIFF_PATH = "data/train/south_america_s2_cloudy/ROIs2017_winter_s2_cloudy_49_p500.tif"
MASK_PATH = "data/train/south_america_s2_masks/ROIs2017_winter_s2_mask_49_p500.tif"
MODEL_PATH = "lama_s2_7in_6out_partial_pretrained.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Cargar modelo
model_s2.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model_s2 = model_s2.to(device)
model_s2.eval()

# Leer TIFF
with rasterio.open(TIFF_PATH) as src:
    img = src.read()  # [C, H, W]

print("Shape TIFF:", img.shape)

# Tomar 6 bandas

BANDS     = [1, 2, 3, 7, 12, 13]   # rasterio usa índices desde 1
img = img[np.array(BANDS)-1].astype(np.float32)

# Normalizar simple
img = img / 10000.0
img = np.clip(img, 0, 1)

with rasterio.open(MASK_PATH) as src:
    mask = src.read(1).astype(np.float32)  # [H, W]

print("Shape máscara:", mask.shape)
print("Valores máscara:", np.unique(mask))

# Si viene 0/255, la paso a 0/1
if mask.max() > 1:
    mask = mask / 255.0

mask = np.clip(mask, 0, 1)
mask = (mask > 0.5).astype(np.float32)
mask = mask[None, :, :]


x_img = torch.from_numpy(img).unsqueeze(0).float()      # [1, 6, H, W]
x_mask = torch.from_numpy(mask).unsqueeze(0).float()    # [1, 1, H, W]

x_img_masked = x_img * (1 - x_mask)  # Enmascaro la imagen con la máscara invertida

x = torch.cat([x_img_masked, x_mask], dim=1).to(device)

with torch.no_grad():
    y = model_s2(x)

print("Input")
print(x.min().item(), x.max().item(), x.mean().item())

print("Output")
print(y.min().item(), y.max().item(), y.mean().item())

print("Salida:", y.shape)


y_vis = y[0].detach().cpu().numpy()
y_vis = np.clip(y_vis, 0, 1)

completed = y.detach().cpu() * x_mask.cpu() + x_img * (1 - x_mask.cpu())

completed_np = completed[0].detach().numpy()
completed_np = np.clip(completed_np, 0, 1)

completed_rgb = np.stack(
    [completed_np[2], completed_np[1], completed_np[0]],
    axis=-1
)


import matplotlib.pyplot as plt


#plot image, mask and output side by side
input_rgb = np.stack([img[2], img[1], img[0]], axis=-1)
output_rgb = np.stack([y_vis[2], y_vis[1], y_vis[0]], axis=-1)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].imshow(input_rgb)
axes[0].set_title("S2 nublada")
axes[0].axis("off")

axes[1].imshow(mask[0], cmap="gray")
axes[1].set_title("Máscara")
axes[1].axis("off")

axes[2].imshow(output_rgb)
axes[2].set_title("Predicción pura")
axes[2].axis("off")

axes[3].imshow(completed_rgb)
axes[3].set_title("Imagen completada")
axes[3].axis("off")

plt.tight_layout()
plt.show()