# """
# Test: carga el modelo SD inpainting, modifica conv_in para 12 canales,
# y hace un forward pass con tu imagen real de SEN12MS-CR.

# Corré esto en tu máquina con:
#     python test_model.py
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import rasterio

# # ─────────────────────────────────────────
# # CONFIG
# # ─────────────────────────────────────────

# TIFF_PATH = "data/south_america_s2_cloudy/ROIs1158_spring_s2_cloudy_17_p136.tif"
# BANDS     = [1, 2, 3, 7, 12, 13]   # rasterio usa índices desde 1
#                                     # → B2, B3, B4, B8, B11, B12
# DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Usando device: {DEVICE}")


# # ─────────────────────────────────────────
# # PASO 1: leer imagen
# # ─────────────────────────────────────────

# from s2cloudless import S2PixelCloudDetector

# # Bandas que necesita s2cloudless (índices rasterio desde 1)
# CLOUD_BANDS_IDX = [1, 2, 4, 5, 8, 9, 10, 11, 12, 13]

# with rasterio.open(TIFF_PATH) as src:
#     # Bandas para s2cloudless
#     cloud_data = src.read(indexes=CLOUD_BANDS_IDX).astype(np.float32) / 10000.0
#     # Bandas que usamos para el modelo
#     data = src.read(indexes=BANDS).astype(np.float32) / 10000.0

# data = np.clip(data, 0, 1)

# # Generar máscara con s2cloudless
# detector   = S2PixelCloudDetector(threshold=0.4, all_bands=False)
# cloud_input = cloud_data.transpose(1, 2, 0)[np.newaxis, ...]  # [1, 256, 256, 10]
# cloud_prob  = detector.get_cloud_probability_maps(cloud_input)[0]
# cloud_mask  = (cloud_prob > 0.4).astype(np.float32)           # [256, 256]

# print(f"  Cobertura de nubes: {cloud_mask.mean()*100:.1f}%")

# # Convertir a tensor [1, 1, 256, 256]
# s2_image = torch.tensor(data).unsqueeze(0).to(DEVICE)
# mask     = torch.tensor(cloud_mask).unsqueeze(0).unsqueeze(0).to(DEVICE)

# print(f"  Tensor imagen: {s2_image.shape}")
# print(f"  Tensor máscara: {mask.shape}")


# DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
from diffusers import StableDiffusionInpaintPipeline

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float32
)
pipe = pipe.to(DEVICE)

vae  = pipe.vae
unet = pipe.unet

print("  Modelo cargado OK")
print(f"  conv_in original: {unet.conv_in}")


# ─────────────────────────────────────────
# PASO 3: modificar conv_in
# ─────────────────────────────────────────

print("\n[3/5] Modificando conv_in para 12 canales...")

old_conv = unet.conv_in
# Conv2d(9, 320, kernel_size=3, padding=1)

NEW_IN_CHANNELS = 12  # 9 originales + 3 bandas extra (B8, B11, B12)

new_conv = nn.Conv2d(
    NEW_IN_CHANNELS,
    old_conv.out_channels,  # 320, no cambia
    kernel_size=3,
    padding=1
).to(DEVICE)

# Copiás los pesos originales en los primeros 9 canales
new_conv.weight.data[:, :9, :, :] = old_conv.weight.data.clone()

# Los 3 canales nuevos arrancan en cero
# → al inicio el modelo se comporta igual que antes del finetuning
new_conv.weight.data[:, 9:, :, :] = 0.0

# Bias sin cambios
new_conv.bias.data = old_conv.bias.data.clone()

# Reemplazás la capa
unet.conv_in = new_conv

print(f"  conv_in modificado: {unet.conv_in}")


# ─────────────────────────────────────────
# PASO 4: preparar inputs
# ─────────────────────────────────────────

print("\n[4/5] Preparando inputs...")

with torch.no_grad():

    # RGB para el VAE (B2, B3, B4 → índices 0, 1, 2 del tensor)
    rgb   = s2_image[:, :3, :, :]   # [1, 3, 256, 256]

    # Bandas extra (B8, B11, B12 → índices 3, 4, 5 del tensor)
    extra = s2_image[:, 3:, :, :]   # [1, 3, 256, 256]

    # SD espera imágenes en rango [-1, 1]
    rgb_normalized = rgb * 2.0 - 1.0

    # Latente de la imagen RGB via VAE
    latente_imagen = vae.encode(rgb_normalized).latent_dist.sample()
    latente_imagen = latente_imagen * vae.config.scaling_factor
    # → [1, 4, 32, 32]

    # Máscara redimensionada al tamaño del latente
    mask_latente = F.interpolate(mask, size=(32, 32), mode="nearest")
    # → [1, 1, 32, 32]

    # Bandas extra redimensionadas al tamaño del latente
    extra_latente = F.interpolate(extra, size=(32, 32), mode="bilinear", align_corners=False)
    # → [1, 3, 32, 32]

    print(f"  latente_imagen: {latente_imagen.shape}")
    print(f"  mask_latente:   {mask_latente.shape}")
    print(f"  extra_latente:  {extra_latente.shape}")


# ─────────────────────────────────────────
# PASO 5: forward pass + decodificar
# ─────────────────────────────────────────

print("\n[5/5] Forward pass...")

import matplotlib.pyplot as plt

with torch.no_grad():

    x_t = torch.randn(1, 4, 32, 32, device=DEVICE)
    t   = torch.tensor([500], device=DEVICE)

    unet_input = torch.cat([
        x_t,
        mask_latente,
        latente_imagen,
        extra_latente
    ], dim=1)

    encoder_hidden_states = torch.zeros(1, 77, 768, device=DEVICE)
    noise_pred = unet(unet_input, t, encoder_hidden_states=encoder_hidden_states).sample

    print(f"  noise_pred shape: {noise_pred.shape}")

    # Decodificar el latente predicho → imagen
    latente_decodificar = noise_pred / vae.config.scaling_factor
    imagen_out = vae.decode(latente_decodificar).sample
    # → [1, 3, 256, 256] en rango [-1, 1]

    # Pasar a [0, 1]
    imagen_out = (imagen_out.clamp(-1, 1) + 1) / 2
    imagen_out = imagen_out[0].permute(1, 2, 0).cpu().numpy()  # [256, 256, 3]

    # RGB input para comparar
    rgb_input = (rgb_normalized.clamp(-1, 1) + 1) / 2
    rgb_input = rgb_input[0].permute(1, 2, 0).cpu().numpy()

    # Plotear
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(rgb_input)
    axes[0].set_title("Input S2 nuboso (RGB)")
    axes[0].axis("off")

    axes[1].imshow(imagen_out)
    axes[1].set_title("Output del modelo (sin finetuning)")
    axes[1].axis("off")

    plt.suptitle("Sin finetuning → output es ruido, es esperado", fontsize=11)
    plt.tight_layout()
    # plt.savefig("test_output.png", dpi=150, bbox_inches="tight")
    plt.show()

print("\n✅ Guardado en test_output.png")