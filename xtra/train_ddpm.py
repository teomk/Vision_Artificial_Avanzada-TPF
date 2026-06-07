import argparse
import math
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diffusers import DDPMScheduler

from xtra.datasets_raro import SatellitePatchDataset
from models.model_ddpm import ConditionalDDPMUNet, ControlNet


def get_scheduler(num_train_timesteps: int = 1000):
    return DDPMScheduler(beta_start=1e-4, beta_end=0.02, beta_schedule="linear", num_train_timesteps=num_train_timesteps)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SatellitePatchDataset(args.data_root, patch_size=args.patch_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    val_dataset = SatellitePatchDataset(args.data_root, patch_size=args.patch_size)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = ConditionalDDPMUNet(image_channels=3, condition_channels=4, base_channels=64, time_dim=256, controlnet=None)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    scheduler = get_scheduler(num_train_timesteps=args.num_timesteps)
    betas = scheduler.betas
    alphas = 1.0 - betas
    alphas_cumprod = torch.from_numpy(alphas.cumprod(axis=0)).float().to(device)

    scaler = torch.cuda.amp.GradScaler()

    global_step = 0
    best_val_loss = float("inf")

    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for batch in dataloader:
            x0 = batch["x0"].to(device)           # [B,C,H,W]
            s2_cloudy = batch["s2_cloudy"].to(device)
            mask = batch["mask"].to(device)
            sar = batch.get("sar")
            if sar is not None:
                sar = sar.to(device)

            B = x0.shape[0]

            t = torch.randint(0, args.num_timesteps, (B,), device=device)
            a_cum = alphas_cumprod[t].view(B, 1, 1, 1)
            sqrt_a_cum = a_cum.sqrt()
            sqrt_one_minus_a_cum = (1 - a_cum).sqrt()

            noise = torch.randn_like(x0)
            x_t = sqrt_a_cum * x0 + sqrt_one_minus_a_cum * noise

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                pred = model(x_t, t, s2_cloudy, mask, sar=None)
                loss = F.mse_loss(pred, noise)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item() * B
            global_step += 1

            if global_step % args.save_every == 0:
                ckpt = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": global_step,
                }
                torch.save(ckpt, Path(args.out_dir) / f"ckpt_{global_step}.pt")

        epoch_loss = epoch_loss / len(dataset)
        t1 = time.time()
        print(f"Epoch {epoch+1}/{args.epochs} loss={epoch_loss:.6f} time={t1-t0:.1f}s")

        # Validation
        val_loss = validate(model, val_loader, alphas_cumprod, device, args)
        print(f"Validation loss: {val_loss:.6f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), Path(args.out_dir) / "best_model.pt")


def validate(model, val_loader, alphas_cumprod, device, args):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in val_loader:
            x0 = batch["x0"].to(device)
            s2_cloudy = batch["s2_cloudy"].to(device)
            mask = batch["mask"].to(device)

            B = x0.shape[0]
            t = torch.randint(0, args.num_timesteps, (B,), device=device)
            a_cum = alphas_cumprod[t].view(B, 1, 1, 1)
            sqrt_a_cum = a_cum.sqrt()
            sqrt_one_minus_a_cum = (1 - a_cum).sqrt()

            noise = torch.randn_like(x0)
            x_t = sqrt_a_cum * x0 + sqrt_one_minus_a_cum * noise

            pred = model(x_t, t, s2_cloudy, mask, sar=None)
            loss = F.mse_loss(pred, noise, reduction="sum")
            total += loss.item()

    return total / (len(val_loader.dataset) * x0.numel())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--out-dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num-timesteps", type=int, default=1000)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--save-every", type=int, default=500)
    args = parser.parse_args()

    train(args)
