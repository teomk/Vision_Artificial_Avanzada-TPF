"""
vram_check.py
=============
Sanity check de VRAM para el modelo DBCR.

Mide:
  - VRAM ocupada por el modelo (parámetros)
  - VRAM ocupada por cada SFBlock durante el forward
  - VRAM pico total del forward
  - Si entra o no en el límite indicado

Uso:
    python vram_check.py                        # prueba bf16 + checkpointing
    python vram_check.py --no-bf16              # prueba fp32 puro
    python vram_check.py --no-checkpoint        # sin gradient checkpointing
    python vram_check.py --vram-limit 24        # cambia el límite (GB)
    python vram_check.py --H 128 --W 128        # resolución distinta
"""

import argparse
import math
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from contextlib import contextmanager


# ─────────────────────────────────────────────
# Utilidades de VRAM
# ─────────────────────────────────────────────

def vram_used_gb():
    """VRAM actualmente reservada en GB."""
    return torch.cuda.memory_reserved() / 1e9


def vram_allocated_gb():
    """VRAM realmente usada (allocated) en GB."""
    return torch.cuda.memory_allocated() / 1e9


def reset_peak():
    torch.cuda.reset_peak_memory_stats()


def peak_vram_gb():
    return torch.cuda.max_memory_allocated() / 1e9


@contextmanager
def measure_block(label, results_dict):
    """Context manager que mide VRAM delta antes/después de un bloque."""
    torch.cuda.synchronize()
    before = vram_allocated_gb()
    reset_peak()
    yield
    torch.cuda.synchronize()
    after  = vram_allocated_gb()
    peak   = peak_vram_gb()
    delta  = after - before
    results_dict[label] = {
        "before_gb": round(before, 3),
        "after_gb":  round(after,  3),
        "delta_gb":  round(delta,  3),
        "peak_gb":   round(peak,   3),
    }


# ─────────────────────────────────────────────
# Arquitectura (copia de model.py con hooks de VRAM)
# ─────────────────────────────────────────────

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device   = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)


class TimeMLP(nn.Module):
    def __init__(self, time_dim):
        super().__init__()
        self.net = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
    def forward(self, t): return self.net(t)


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias   = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps    = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var  = x.var(dim=1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class TimeNAFBlock(nn.Module):
    def __init__(self, channels, time_dim, dw_expand=2, ffn_expand=2):
        super().__init__()
        dw_ch  = channels * dw_expand
        ffn_ch = channels * ffn_expand
        self.norm1      = LayerNorm2d(channels)
        self.conv1      = nn.Conv2d(channels, dw_ch, 1)
        self.dwconv     = nn.Conv2d(dw_ch, dw_ch, 3, padding=1, groups=dw_ch)
        self.sg         = SimpleGate()
        self.sca        = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                        nn.Conv2d(dw_ch//2, dw_ch//2, 1))
        self.conv2      = nn.Conv2d(dw_ch//2, channels, 1)
        self.time_proj1 = nn.Linear(time_dim, channels)
        self.norm2      = LayerNorm2d(channels)
        self.conv3      = nn.Conv2d(channels, ffn_ch, 1)
        self.conv4      = nn.Conv2d(ffn_ch//2, channels, 1)
        self.time_proj2 = nn.Linear(time_dim, channels)
        self.beta       = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma      = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = self.conv1(h); h = self.dwconv(h); h = self.sg(h)
        h = h * self.sca(h); h = self.conv2(h)
        h = h + self.time_proj1(t_emb)[:, :, None, None]
        x = x + self.beta * h
        h = self.norm2(x); h = self.conv3(h); h = self.sg(h)
        h = self.conv4(h) + self.time_proj2(t_emb)[:, :, None, None]
        return x + self.gamma * h


class SFBlock(nn.Module):
    """Cross-modal attention SAR↔optical.
    Soporta attention global (paper original) o window attention (fallback).
    """
    def __init__(self, channels, num_heads=1, window_size=None):
        super().__init__()
        self.num_heads   = num_heads
        self.scale       = (channels // num_heads) ** -0.5
        self.window_size = window_size   # None → global (paper original)
        self.to_q    = nn.Conv2d(channels, channels, 1)
        self.to_k    = nn.Conv2d(channels, channels, 1)
        self.to_v    = nn.Conv2d(channels, channels, 1)
        self.out_conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm_opt = LayerNorm2d(channels)
        self.norm_sar = LayerNorm2d(channels)

    def _attn_global(self, Q, K, V, B, C, H, W):
        hd = C // self.num_heads
        Q = Q.reshape(B, self.num_heads, hd, H*W).permute(0,1,3,2)
        K = K.reshape(B, self.num_heads, hd, H*W).permute(0,1,3,2)
        V = V.reshape(B, self.num_heads, hd, H*W).permute(0,1,3,2)
        out = F.scaled_dot_product_attention(Q, K, V)
        return out.permute(0,1,3,2).reshape(B, C, H, W)

    def _attn_window(self, Q, K, V, B, C, H, W):
        ws     = self.window_size
        hd     = C // self.num_heads

        # Alinear K y V a la resolución de Q si SAR y optical tienen distinto tamaño
        if K.shape[-2:] != (H, W):
            K = F.interpolate(K, size=(H, W), mode="bilinear", align_corners=False)
            V = F.interpolate(V, size=(H, W), mode="bilinear", align_corners=False)

        nH, nW = H // ws, W // ws

        def partition(t):
            # t: [B, C, H, W]
            t = t.reshape(B, C, nH, ws, nW, ws)
            t = t.permute(0, 2, 4, 3, 5, 1).contiguous()  # [B, nH, nW, ws, ws, C]
            t = t.reshape(B * nH * nW, ws * ws, self.num_heads, hd)
            return t.permute(0, 2, 1, 3)                   # [B*nH*nW, heads, tokens, hd]

        Q, K, V = partition(Q), partition(K), partition(V)
        out = F.scaled_dot_product_attention(Q, K, V)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.reshape(B * nH * nW, ws, ws, C)
        out = out.reshape(B, nH, nW, ws, ws, C)
        out = out.permute(0, 5, 1, 3, 2, 4).contiguous()
        return out.reshape(B, C, H, W)

    def forward(self, optical, sar):
        B, C, H, W = optical.shape
        opt_n = self.norm_opt(optical)
        sar_n = self.norm_sar(sar)
        Q = self.to_q(opt_n)
        K = self.to_k(sar_n)
        V = self.to_v(sar_n)
        if self.window_size is not None and H > self.window_size:
            out = self._attn_window(Q, K, V, B, C, H, W)
        else:
            out = self._attn_global(Q, K, V, B, C, H, W)
        return optical + self.out_conv(out)


class DownBlockNAF(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.naf  = TimeNAFBlock(out_ch, time_dim)
        self.down = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x, t_emb):
        x    = self.proj(x)
        x    = self.naf(x, t_emb)
        skip = x
        return self.down(x), skip


class UpBlockNAF(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, time_dim):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        in_block  = out_ch + skip_ch
        self.proj = nn.Conv2d(in_block, out_ch, 1) if in_block != out_ch else nn.Identity()
        self.naf  = TimeNAFBlock(out_ch, time_dim)

    def forward(self, x, skip, t_emb):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.proj(x)
        return self.naf(x, t_emb)


class SAREncoderBranch(nn.Module):
    def __init__(self, sar_ch=2, base_ch=64, time_dim=256):
        super().__init__()
        C = base_ch
        self.in_conv = nn.Conv2d(sar_ch, C, 3, padding=1)
        self.down1   = DownBlockNAF(C,     C*2, time_dim)
        self.down2   = DownBlockNAF(C*2,   C*4, time_dim)
        self.down3   = DownBlockNAF(C*4,   C*8, time_dim)
        self.mid     = TimeNAFBlock(C*8, time_dim)

    def forward(self, sar, t_emb):
        # scale0: full resolution, C canales
        x0 = self.in_conv(sar)              # [B, C, H, W]

        # scale1: H/2, C*2 canales
        x1, _ = self.down1(x0, t_emb)       # [B, C*2, H/2, W/2]

        # scale2: H/4, C*4 canales
        x2, _ = self.down2(x1, t_emb)       # [B, C*4, H/4, W/4]

        # scale3: H/8, C*8 canales
        x3, _ = self.down3(x2, t_emb)       # [B, C*8, H/8, W/8]

        # bottleneck: H/8, C*8 canales
        mid = self.mid(x3, t_emb)

        return {
            "scale0": x0,
            "scale1": x1,
            "scale2": x2,
            "scale3": x3,
            "mid": mid,
        }


class DBCR(nn.Module):
    def __init__(self, image_ch=6, cond_ch=6, sar_ch=2,
                 base_ch=64, time_dim=256, num_heads=1,
                 use_checkpoint=False, window_size=None, window_size_sf0=None):
        super().__init__()
        C = base_ch
        self.use_checkpoint = use_checkpoint

        self.time_mlp   = TimeMLP(time_dim)
        self.sar_branch = SAREncoderBranch(sar_ch, C, time_dim)
        self.init_conv  = nn.Conv2d(image_ch + cond_ch, C, 3, padding=1)

        sf = lambda ch: SFBlock(ch, num_heads, window_size)
        sf0 = lambda ch: SFBlock(ch, num_heads, window_size_sf0)
        self.sf0    = sf0(C)
        self.down1  = DownBlockNAF(C,   C*2, time_dim)
        self.sf1    = sf(C*2)
        self.down2  = DownBlockNAF(C*2, C*4, time_dim)
        self.sf2    = sf(C*4)
        self.down3  = DownBlockNAF(C*4, C*8, time_dim)
        self.sf3    = sf(C*8)
        self.mid1   = TimeNAFBlock(C*8, time_dim)
        self.mid2   = TimeNAFBlock(C*8, time_dim)
        self.sf_mid = sf(C*8)
        self.up3    = UpBlockNAF(C*8, C*8, C*4, time_dim)
        self.up2    = UpBlockNAF(C*4, C*4, C*2, time_dim)
        self.up1    = UpBlockNAF(C*2, C*2, C,   time_dim)
        self.out    = nn.Conv2d(C, image_ch, 3, padding=1)

    def _ckpt(self, fn, *args):
        """Aplica gradient checkpoint si está habilitado."""
        if self.use_checkpoint:
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def forward(self, x_t, t, s2_cloudy, sar, vram_log=None):
        """
        vram_log: dict opcional donde se guardan mediciones por bloque.
        """
        def log(label):
            if vram_log is not None:
                torch.cuda.synchronize()
                vram_log[label] = round(vram_allocated_gb(), 3)

        t_emb     = self.time_mlp(t)
        log("after_time_emb")

        sar_feats = self.sar_branch(sar, t_emb)
        log("after_sar_branch")

        x = self.init_conv(torch.cat([x_t, s2_cloudy], dim=1))
        log("after_init_conv")

        x = self._ckpt(self.sf0, x, sar_feats["scale0"])
        log("after_sf0  ← resolución full")

        x, skip1 = self.down1(x, t_emb)
        x = self._ckpt(self.sf1, x, sar_feats["scale1"])
        log("after_sf1  ← H/2")

        x, skip2 = self.down2(x, t_emb)
        x = self._ckpt(self.sf2, x, sar_feats["scale2"])
        log("after_sf2  ← H/4")

        x, skip3 = self.down3(x, t_emb)
        x = self._ckpt(self.sf3, x, sar_feats["scale3"])
        log("after_sf3  ← H/8")

        x = self._ckpt(self.mid1, x, t_emb)
        x = self._ckpt(self.mid2, x, t_emb)
        x = self._ckpt(self.sf_mid, x, sar_feats["mid"])
        log("after_sf_mid (bottleneck)")

        x = self.up3(x, skip3, t_emb)
        x = self.up2(x, skip2, t_emb)
        x = self.up1(x, skip1, t_emb)
        log("after_decoder")

        return self.out(x)


# ─────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────

def run_check(args):
    if not torch.cuda.is_available():
        print("❌  No hay GPU disponible.")
        return

    device    = torch.device("cuda")
    dtype     = torch.bfloat16 if args.bf16 else torch.float32
    dtype_str = "bfloat16" if args.bf16 else "float32"

    print(f"\n{'='*60}")
    print(f"  GPU          : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM total   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Límite check : {args.vram_limit} GB")
    print(f"  Resolución   : {args.H}×{args.W}")
    print(f"  Dtype        : {dtype_str}")
    print(f"  Checkpoint   : {args.checkpoint}")
    print(f"  Window size  : {args.window_size if args.window_size else 'None (global attention)'}")
    print(f"{'='*60}\n")

    results = {}
    success = True
    error   = None

    try:
        # ── 1. Modelo en GPU ───────────────────────────────────────
        torch.cuda.empty_cache(); gc.collect()
        reset_peak()

        with measure_block("1. modelo_en_gpu", results):
            model = DBCR(
                image_ch=6, cond_ch=6, sar_ch=2,
                base_ch=args.base_ch, time_dim=args.time_dim,
                num_heads=args.num_heads,
                use_checkpoint=args.checkpoint,
                window_size=None,
                window_size_sf0=args.window_size,
            ).to(device=device, dtype=dtype)

        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  Parámetros   : {n_params:.2f}M")
        print(f"  VRAM modelo  : {results['1. modelo_en_gpu']['delta_gb']:.3f} GB\n")

        # ── 2. Inputs ──────────────────────────────────────────────
        B, H, W = args.batch, args.H, args.W
        with measure_block("2. inputs_en_gpu", results):
            x_t       = torch.randn(B, 6, H, W, device=device, dtype=dtype)
            t         = torch.randint(0, 1000, (B,), device=device)
            s2_cloudy = torch.randn(B, 6, H, W, device=device, dtype=dtype)
            sar       = torch.randn(B, 2, H, W, device=device, dtype=dtype)

        print(f"  VRAM inputs  : {results['2. inputs_en_gpu']['delta_gb']:.3f} GB\n")

        # ── 3. Forward con log por bloque ─────────────────────────
        vram_log = {}
        reset_peak()

        ctx = torch.autocast(device_type="cuda", dtype=dtype) if args.bf16 else torch.no_grad()

        print("  Forward pass — VRAM por bloque (allocated):")
        print(f"  {'Bloque':<35} {'VRAM (GB)':>10}")
        print(f"  {'-'*45}")

        with ctx:
            if args.backward:
                out = model(x_t, t, s2_cloudy, sar, vram_log=vram_log)
                loss = out.mean()
                loss.backward()
            else:
                with torch.no_grad():
                    out = model(x_t, t, s2_cloudy, sar, vram_log=vram_log)

        for k, v in vram_log.items():
            print(f"  {k:<35} {v:>10.3f}")

        peak = peak_vram_gb()
        print(f"\n  {'PICO total (peak allocated)':<35} {peak:>10.3f} GB")

        results["peak_gb"]  = peak
        results["output"]   = out.shape

    except torch.cuda.OutOfMemoryError as e:
        success = False
        error   = str(e)
        peak    = torch.cuda.max_memory_allocated() / 1e9

    # ── Veredicto ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    if success:
        limit = args.vram_limit
        peak  = results.get("peak_gb", 0)
        margen = limit - peak
        emoji  = "✅" if margen >= 0 else "⚠️ "
        print(f"  {emoji}  VRAM pico    : {peak:.3f} GB  /  límite: {limit} GB")
        print(f"       Margen       : {margen:+.3f} GB")
        print(f"       Output shape : {results.get('output', 'N/A')}")
        if margen < 0:
            print("\n  ❌  No entra. Opciones:")
            print("      · Probar window_size (--window-size 8)")
            print("      · Reducir batch size")
            print("      · Reducir base_ch (actualmente", args.base_ch, ")")
    else:
        print(f"  ❌  OOM durante el forward.")
        print(f"       VRAM al momento del crash : {peak:.3f} GB")
        print(f"       Error: {error[:120]}")
        print("\n  Opciones para intentar:")
        print("      · Probar window attention  : --window-size 8")
        print("      · Habilitar bf16           : (ya activado si usaste --bf16)")
        print("      · Habilitar checkpointing  : (ya activado si usaste --checkpoint)")
        print("      · Reducir resolución       : --H 128 --W 128")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DBCR VRAM sanity check")

    # Resolución / batch
    parser.add_argument("--H",           type=int,   default=256)
    parser.add_argument("--W",           type=int,   default=256)
    parser.add_argument("--batch",       type=int,   default=1)

    # Arquitectura
    parser.add_argument("--base-ch",     type=int,   default=64)
    parser.add_argument("--time-dim",    type=int,   default=128)
    parser.add_argument("--num-heads",   type=int,   default=1)
    parser.add_argument("--window-size", type=int,   default=None,
                        help="Window size para SFBlock (None = global attention del paper)")

    # Optimizaciones
    parser.add_argument("--no-bf16",       action="store_true",
                        help="Usar fp32 en vez de bfloat16")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="Desactivar gradient checkpointing")

    # Test de backward también
    parser.add_argument("--backward",    action="store_true",
                        help="Medir VRAM incluyendo el backward pass (training)")

    # Límite
    parser.add_argument("--vram-limit",  type=float, default=24,
                        help="VRAM disponible en GB (default: 24)")

    args = parser.parse_args()
    args.bf16       = not args.no_bf16
    args.checkpoint = not args.no_checkpoint

    run_check(args)