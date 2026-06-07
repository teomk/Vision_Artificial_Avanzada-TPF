import torch
import sys
from pathlib import Path
import yaml
import argparse

ROOT = Path(__file__).resolve().parent.parent
LAMA_DIR = ROOT / "external" / "lama"
sys.path.append(str(LAMA_DIR))

from saicinpainting.training.modules.ffc import FFCResNetGenerator


# ── Adaptadores de primera conv ────────────────────────────────────────

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
    adapted = adapt_conv_4_to_7(old_weight, new_weight[:, :7])  # reutiliza lógica anterior

    rgb_mean = old_weight[:, 0:3].mean(dim=1, keepdim=True)
    s1_vv = rgb_mean
    s1_vh = rgb_mean

    adapted = torch.cat([adapted, s1_vv, s1_vh], dim=1).float()         # [out, 9, k, k]
    return adapted


def adapt_conv_7_to_9(old_weight, new_weight):
    """S2 finetuned (7 canales) → S2 con SAR (9 canales): agrega 2 canales SAR al final"""
    s2_mean = old_weight[:, 0:6].mean(dim=1, keepdim=True)
    s1_vv   = s2_mean
    s1_vh   = s2_mean
    return torch.cat([old_weight, s1_vv, s1_vh], dim=1).float()         # [out, 9, k, k]


# ── Función genérica de adaptación ────────────────────────────────────

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
    # python models/lama_model.py --config configs/lama_no_sar.yaml
    # python models/lama_model.py --config configs/lama_sar.yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path al archivo yaml de configuración")
    # Overrides opcionales desde CLI
    parser.add_argument("--sar",              action="store_true", default=None)
    parser.add_argument("--pretrained_path",  type=str,            default=None)
    parser.add_argument("--save_name",        type=str,            default=None)
    args = parser.parse_args()

    # Cargar yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    cfg_adapt = cfg["adapt"]

    # CLI overrides yaml si se especifica
    if args.sar:
        cfg["sar"] = True
    if args.pretrained_path:
        cfg_adapt["pretrained_path"] = args.pretrained_path
    if args.save_name:
        cfg_adapt["save_name"] = args.save_name

    # Usar cfg desde acá
    use_sar   = cfg.get("sar", False)
    input_nc  = 9 if use_sar else 7

    pretrained = cfg_adapt.get("pretrained_path")
    CKPT_PATH  = Path(pretrained) if pretrained else ROOT / "big-lama" / "models" / "best.ckpt"

    save_name  = cfg_adapt.get("save_name", "lama_pretrained.pth")
    save_path  = ROOT / "saved_models" / save_name

    if save_path.exists():
        print(f"El archivo de destino ya existe: {save_path}")
        overwrite = input("¿Desea sobrescribirlo? (s/n): ").strip().lower()
        if overwrite != 's':
            print("Operación cancelada por el usuario.")
            sys.exit(0)

    if not save_path.parent.exists():
        save_path.parent.mkdir(parents=True, exist_ok=True)

    

    print(f"\nModo     : {'con SAR (9 canales)' if use_sar else 'sin SAR (7 canales)'}")
    print(f"Guardando: {save_path}\n")

    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"No se encontró el checkpoint en: {CKPT_PATH}")

    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    model = adapt_model(state_dict, input_nc=input_nc, output_nc=6)
    model = model.float()

    torch.save(model.state_dict(), str(save_path))
    print(f"\nModelo guardado en: {save_path}")