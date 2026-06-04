Quick training scaffold for conditional DDPM inpainting.

Setup
```
python -m venv .venv
pip install -r requirements.txt
```

Run training (example):
```bash
python train.py --data-root data --out-dir checkpoints --epochs 20 --batch-size 8 --patch-size 128
```

Notes:
- Put your patched dataset folders under `data/` named `south_america_s2`, `south_america_s2_cloudy`, `south_america_s1`.
- `train.py` uses a Hugging Face `DDPMScheduler` for the noise schedule.
- Augmentations happen on-the-fly in `datasets.py`.
