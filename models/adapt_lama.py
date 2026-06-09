import torch
import sys
from pathlib import Path
import yaml
import argparse

ROOT     = Path(__file__).resolve().parent.parent
LAMA_DIR = ROOT / "external" / "lama"
sys.path.append(str(LAMA_DIR))
sys.path.append(str(ROOT / "utils"))

from saicinpainting.training.modules.ffc import FFCResNetGenerator
from hf_utils import (
    download_model,
    upload_model,
    resolve_load_version,
    resolve_save_version,
)


# ── Adaptadores de pesos ───────────────────────────────────────────────

def adapt_conv_4_to_7(old_weight, new_weight):
    """big-lama (4 canales: RGB+mask) → S2 sin SAR (7 canales: 6 bandas S2 + mask)"""
    adapted = new_weight.clone()

    old_r    = old_weight[:, 0:1]
    old_g    = old_weight[:, 1:2]
    old_b    = old_weight[:, 2:3]
    old_mask = old_weight[:, 3:4]
    rgb_mean = old_weight[:, 0:3].mean(dim=1, keepdim=True)

    adapted[:, 0:1] = old_b       # B2 azul
    adapted[:, 1:2] = old_g       # B3 verde
    adapted[:, 2:3] = old_r       # B4 rojo
    adapted[:, 3:4] = rgb_mean    # B8 NIR
    adapted[:, 4:5] = rgb_mean    # B11 SWIR
    adapted[:, 5:6] = rgb_mean    # B12 SWIR
    adapted[:, 6:7] = old_mask    # máscara

    return adapted.float()

def adapt_conv_4_to_9(old_weight, new_weight):
    """big-lama (4 canales: RGB+mask) → S2 con SAR (9 canales: 6 bandas S2 + mask + 2 SAR)"""
    adapted  = adapt_conv_4_to_7(old_weight, new_weight[:, :7])
    rgb_mean = old_weight[:, 0:3].mean(dim=1, keepdim=True)
    return torch.cat([adapted, rgb_mean, rgb_mean], dim=1).float()  # [out, 9, k, k]

def adapt_conv_7_to_9(old_weight, new_weight):
    """S2 finetuned (7 canales) → S2 con SAR (9 canales): agrega 2 canales SAR al final"""
    s2_mean = old_weight[:, 0:6].mean(dim=1, keepdim=True)
    return torch.cat([old_weight, s2_mean, s2_mean], dim=1).float()  # [out, 9, k, k]


# ── Adaptación del modelo ──────────────────────────────────────────────

def adapt_model(ckpt_state, input_nc, output_nc):
    model = FFCResNetGenerator(
        input_nc=input_nc,
        output_nc=output_nc,
        ngf=64,
        n_downsampling=3,
        n_blocks=18,
        init_conv_kwargs      ={"ratio_gin": 0,    "ratio_gout": 0},
        downsample_conv_kwargs={"ratio_gin": 0,    "ratio_gout": 0},
        resnet_conv_kwargs    ={"ratio_gin": 0.75, "ratio_gout": 0.75, "enable_lfu": False},
    )

    model_dict    = model.state_dict()
    new_state     = {}
    loaded        = 0
    reinitialized = 0

    for name, weight in model_dict.items():
        # El checkpoint big-lama tiene prefijo "generator.", los .pth no
        ckpt_name = "generator." + name if ("generator." + name) in ckpt_state else name

        if ckpt_name not in ckpt_state:
            new_state[name] = weight
            reinitialized += 1
            print(f"No existe, se reinicializa : {name} {weight.shape}")
            continue

        old_weight = ckpt_state[ckpt_name]

        # Shapes exactos → carga directa
        if old_weight.shape == weight.shape:
            new_state[name] = old_weight
            loaded += 1

        # 4 → 7
        elif (name == "model.1.ffc.convl2l.weight"
              and old_weight.shape[1] == 4 and weight.shape[1] == 7):
            print(f"Adaptando conv 4→7 : {old_weight.shape} → {weight.shape}")
            new_state[name] = adapt_conv_4_to_7(old_weight, weight)
            loaded += 1

        # 4 → 9
        elif (name == "model.1.ffc.convl2l.weight"
              and old_weight.shape[1] == 4 and weight.shape[1] == 9):
            print(f"Adaptando conv 4→9 : {old_weight.shape} → {weight.shape}")
            new_state[name] = adapt_conv_4_to_9(old_weight, weight)
            loaded += 1

        # 7 → 9
        elif (name == "model.1.ffc.convl2l.weight"
              and old_weight.shape[1] == 7 and weight.shape[1] == 9):
            print(f"Adaptando conv 7→9 : {old_weight.shape} → {weight.shape}")
            new_state[name] = adapt_conv_7_to_9(old_weight, weight)
            loaded += 1

        else:
            new_state[name] = weight
            reinitialized += 1
            print(f"Incompatible, reinicializa : {name} ckpt={old_weight.shape} nuevo={weight.shape}")

    model.load_state_dict(new_state, strict=False)
    print(f"\nPesos cargados      : {loaded}")
    print(f"Pesos reinicializados: {reinitialized}")
    return model


# ── Main ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Ejemplos de uso:
    #   Sin SAR (desde big-lama local → sube como no_sar pretrained):
    #     python models/adapt_lama.py --config configs/lama_no_sar.yaml
    #     python models/adapt_lama.py --config configs/adapt_lama_no_sar.yaml --save_version 2
    #
    #   Con SAR (baja último finetuned no_sar → sube como sar pretrained):
    #     python models/adapt_lama.py --config configs/lama_sar.yaml
    #     python models/adapt_lama.py --config configs/lama_sar.yaml --load_version 3
    #     python models/adapt_lama.py --config configs/adapt_lama_sar.yaml --load_version 3 --save_version 2

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       type=str, required=True,
                        help="Path al archivo yaml de configuración")
    parser.add_argument("--load_version", type=int, default=None,
                        help="[Solo SAR] Versión del finetuned no-SAR a cargar. "
                             "Si no se indica, usa la última disponible.")
    parser.add_argument("--save_version", type=int, default=None,
                        help="Versión con la que guardar el pretrained resultante. "
                             "Si no se indica, auto-incrementa.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    cfg_adapt = cfg["adapt"]
    cfg_hf    = cfg["huggingface"]

    use_sar  = cfg.get("sar", False)
    input_nc = 9 if use_sar else 7

    repo_id              = cfg_hf["repo_id"]               # e.g. "LucioLuque/lama"
    no_sar_fine_prefix   = cfg_hf["no_sar_fine_prefix"]    # e.g. "lama_no_sar_finetuned"
    save_prefix          = cfg_hf["adapt_save_prefix"]     # e.g. "lama_no_sar_pretrained"
                                                           #   o  "lama_sar_pretrained"

    print(f"\nModo: {'con SAR (9 canales)' if use_sar else 'sin SAR (7 canales)'}")

    # ── Cargar checkpoint fuente ──────────────────────────────────────────────
    if use_sar:
        # Baja el finetuned no-SAR desde HuggingFace (último o el especificado)
        _, load_filename = resolve_load_version(
            repo_id=repo_id,
            filename_prefix=no_sar_fine_prefix,
            requested_version=args.load_version,
        )
        print(f"Cargando desde HuggingFace: {load_filename}")
        ckpt_state = download_model(repo_id=repo_id, filename=load_filename, map_location="cpu")

    else:
        # Carga el big-lama local
        ckpt_path = Path(cfg_adapt.get("pretrained_path",
                         str(ROOT / "big-lama" / "models" / "best.ckpt")))
        if not ckpt_path.exists():
            raise FileNotFoundError(f"No se encontró el checkpoint en: {ckpt_path}")
        print(f"Cargando desde disco: {ckpt_path}")
        ckpt      = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ckpt_state = ckpt.get("state_dict", ckpt)

    # ── Adaptar modelo ────────────────────────────────────────────────────────
    model = adapt_model(ckpt_state, input_nc=input_nc, output_nc=6)
    model = model.float()

    # ── Resolver versión de guardado ──────────────────────────────────────────
    save_version, save_filename = resolve_save_version(
        repo_id=repo_id,
        filename_prefix=save_prefix,
        requested_version=args.save_version,
    )

    # ── Subir modelo a HuggingFace ────────────────────────────────────────────
    upload_model(
        model_state_dict=model.state_dict(),
        repo_id=repo_id,
        filename=save_filename,
    )