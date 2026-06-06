import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LAMA_DIR = ROOT / "external" / "lama"

sys.path.append(str(LAMA_DIR))


from saicinpainting.training.modules.ffc import FFCResNetGenerator


device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Modelo nuevo para Sentinel-2
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
CKPT_PATH = ROOT / "big-lama" / "models" / "best.ckpt"

if not CKPT_PATH.exists():
    raise FileNotFoundError(f"No se encontró el checkpoint en: {CKPT_PATH}")

ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)# 2. Cargar checkpoint big-lama
# ckpt = torch.load("big-lama/models/best.ckpt", map_location="cpu")

def adapt_first_conv(old_weight, new_weight):
    """
    old_weight: [out_channels, 4, k, k]  -> RGB + mask
    new_weight: [out_channels, 7, k, k]  -> 6 bandas S2 + mask
    """

    adapted = new_weight.clone()

    old_r = old_weight[:, 0:1]
    old_g = old_weight[:, 1:2]
    old_b = old_weight[:, 2:3]
    old_mask = old_weight[:, 3:4]

    rgb_mean = old_weight[:, 0:3].mean(dim=1, keepdim=True)

    # Supongamos orden S2: [B2, B3, B4, B8, B11, B12, mask]
    adapted[:, 0:1] = old_b       # B2 azul
    adapted[:, 1:2] = old_g       # B3 verde
    adapted[:, 2:3] = old_r       # B4 rojo
    adapted[:, 3:4] = rgb_mean    # B8 NIR
    adapted[:, 4:5] = rgb_mean    # B11 SWIR
    adapted[:, 5:6] = rgb_mean    # B12 SWIR
    adapted[:, 6:7] = old_mask    # máscara

    return adapted


# Según el checkpoint, puede estar en una de estas claves
if "state_dict" in ckpt:
    state_dict = ckpt["state_dict"]
else:
    state_dict = ckpt

model_dict = model_s2.state_dict()

new_state_dict = {}

loaded = 0
reinitialized = 0

for name, weight in model_dict.items():

    ckpt_name = "generator." + name

    if ckpt_name in state_dict:
        old_weight = state_dict[ckpt_name]

        # Caso 1: pesos compatibles exactos
        if old_weight.shape == weight.shape:
            new_state_dict[name] = old_weight
            loaded += 1

        # Caso 2: primera conv, de 4 canales a 7 canales
        elif (
            len(old_weight.shape) == 4
            and len(weight.shape) == 4
            and old_weight.shape[1] == 4
            and weight.shape[1] == 7
            and old_weight.shape[0] == weight.shape[0]
            and old_weight.shape[2:] == weight.shape[2:]
        ):
            print(f"Adaptando primera conv: {name} | {old_weight.shape} -> {weight.shape}")
            new_state_dict[name] = adapt_first_conv(old_weight, weight)
            loaded += 1

        # Caso 3: incompatible
        else:
            new_state_dict[name] = weight
            reinitialized += 1
            print(f"No compatible, se reinicializa: {name} | ckpt {old_weight.shape} -> nuevo {weight.shape}")

    else:
        new_state_dict[name] = weight
        reinitialized += 1
        print(f"No existe en checkpoint, se reinicializa: {name} | nuevo {weight.shape}")

print(f"Pesos cargados: {loaded}")
print(f"Pesos reinicializados: {reinitialized}")

model_s2.load_state_dict(new_state_dict, strict=False)


torch.save(model_s2.state_dict(), "lama_s2_7in_6out_partial_pretrained.pth")