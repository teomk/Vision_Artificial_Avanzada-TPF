from huggingface_hub import hf_hub_download
import torch
# eval/visualize_lama.py
from pathlib import Path
import torch
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yaml
import argparse
from torch.utils.data import DataLoader

ROOT     = Path(__file__).resolve().parent.parent
LAMA_DIR = ROOT / "external" / "lama"
DATA_DIR = ROOT / "dataset"
sys.path.append(str(LAMA_DIR))
sys.path.append(str(DATA_DIR))

from saicinpainting.training.modules.ffc import FFCResNetGenerator
from dataset_lama import SEN12MSCRDataset

model_path = hf_hub_download(
    repo_id="LucioLuque/lama_no_sar",
    filename="lama_no_sar_finetuned_v1.pth"
)

checkpoint = torch.load(model_path, map_location="cpu")


def build_model(use_sar: bool) -> FFCResNetGenerator:
    return FFCResNetGenerator(
        input_nc  = 9 if use_sar else 7,
        output_nc = 6,
        ngf       = 64,
        n_downsampling = 3,
        n_blocks   = 18,
        init_conv_kwargs      ={"ratio_gin": 0,    "ratio_gout": 0},
        downsample_conv_kwargs={"ratio_gin": 0,    "ratio_gout": 0},
        resnet_conv_kwargs    ={"ratio_gin": 0.75, "ratio_gout": 0.75, "enable_lfu": False},
    )


model = build_model(use_sar=False)
model.load_state_dict(checkpoint)
print("todo ok!")