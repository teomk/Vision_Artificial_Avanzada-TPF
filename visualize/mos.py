import tkinter as tk
from tkinter import ttk
import json
import random
import numpy as np
import torch
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageTk

ROOT       = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
UTILS_DIR  = ROOT / "utils"
DATA_DIR   = ROOT / "dataset"

sys.path.append(str(DATA_DIR))
sys.path.append(str(MODELS_DIR))
sys.path.append(str(UTILS_DIR))

from dbcr_simple import DBCRSimple
from dbcr_complex import DBCR
from hf_utils import download_model
from dbcr_utils import inference
from dataset_utils import unpack_batch
from evaluate_utils import psnr

S2_PATH      = ROOT / "visualize" / "s2_6bands.npy"
S1_PATH      = ROOT / "visualize" / "s1_2bands.npy"
RESULTS_PATH = ROOT / "visualize" / "mos_results.json"

REPO_ID   = "LucioLuque/lama"
T         = 1000
STEPS     = 1
SIGMOID_K = 10.0

S1_MEAN = np.array([-8.999908447265625, -14.78221321105957], dtype=np.float32)
S1_STD  = np.array([2.413282871246338,  2.3029115200042725], dtype=np.float32)

IMG_SIZE = 256

# ── Normalización ─────────────────────────────────────────────────────

def normalize_s2(s2_raw):
    return np.clip(s2_raw / 10000.0, 0, 1).astype(np.float32)

def normalize_s1(s1_raw):
    mean = S1_MEAN[:, None, None]
    std  = S1_STD[:, None, None]
    return ((s1_raw - mean) / (std + 1e-6)).astype(np.float32)

def to_rgb_np(tensor_chw, bands=(2, 1, 0)):
    arr = tensor_chw[list(bands)].transpose(1, 2, 0)
    p2, p98 = np.percentile(arr, 2), np.percentile(arr, 98)
    arr = np.clip((arr - p2) / (p98 - p2 + 1e-8), 0, 1)
    return (arr * 255).astype(np.uint8)

def to_sar_np(s1_raw, band=0):
    img = s1_raw[band]
    p2, p98 = np.percentile(img, 2), np.percentile(img, 98)
    img = np.clip((img - p2) / (p98 - p2 + 1e-8), 0, 1)
    return (img * 255).astype(np.uint8)

# ── Carga de modelos ──────────────────────────────────────────────────

def load_models(device):
    print("Cargando modelos...")

    def load_simple(filename, sar_mode):
        condition_channels = 8 if sar_mode == "Concat" else 6
        ckpt  = download_model(repo_id=REPO_ID, filename=filename, map_location=device)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model = DBCRSimple(image_channels=6, condition_channels=condition_channels,
                           base_channels=64, time_dim=128, control_net=(sar_mode == "ControlNet"))
        model.load_state_dict(state, strict=False)
        return model.float().to(device).eval()

    def load_complex(filename):
        ckpt  = download_model(repo_id=REPO_ID, filename=filename, map_location=device)
        model = DBCR(image_channels=6, condition_channels=6, sar_channels=2,
                     base_channels=64, time_dim=128, num_heads=1,
                     window_size_sf0=8, window_size_not_sf0=None,
                     use_checkpoint=True, include_encoder_4=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        return model.float().to(device).eval()

    return {
        "DBCR-S":           (load_simple("dbcr_no_sar_naf_v2.pth", "None"),       "None",        "simple"),
        "DBCR-SC (sin TL)": (load_simple("dbcr_concat_v2.pth",     "Concat"),     "Concat",      "simple"),
        "DBCR":             (load_complex("dbcr_complex_v4.pth"),                  "ControlNet",  "complex"),
    }

def run_inference(model, model_type, sar_mode, cloudy_t, s1_t, device):
    cloudy_b = cloudy_t.unsqueeze(0).float().to(device)
    if model_type == "simple":
        if sar_mode == "None":
            condition, sar = cloudy_b, None
        elif sar_mode == "Concat":
            s1_b      = s1_t.unsqueeze(0).float().to(device)
            condition = torch.cat([cloudy_b, s1_b], dim=1)
            sar       = None
        else:
            s1_b      = s1_t.unsqueeze(0).float().to(device)
            condition = cloudy_b
            sar       = s1_b
        pred = inference(model, cloudy_b, condition, device,
                         T=T, steps=STEPS, sar=sar, sigmoid_k=SIGMOID_K, show_progress=False)
    else:
        fake_batch            = (s1_t.unsqueeze(0).float(), cloudy_b, cloudy_b)
        s2_cloudy, _, cond, sar = unpack_batch(fake_batch, sar_mode, device)
        pred = inference(model, s2_cloudy, cond, device,
                         T=T, steps=STEPS, sar=sar, sigmoid_k=SIGMOID_K)
    return pred.squeeze(0).clamp(0, 1).cpu()

# ── Resultados ────────────────────────────────────────────────────────

def load_results():
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {"votes": {}, "sessions": []}

def save_results(results):
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

# ── App ───────────────────────────────────────────────────────────────

class MOSApp:
    def __init__(self, root, models, device):
        self.root    = root
        self.models  = models
        self.device  = device
        self.results = load_results()

        self.s2_raw = np.load(S2_PATH)   # [6, 256, 256] crudo
        self.s1_raw = np.load(S1_PATH)   # [2, 256, 256] crudo

        self.s2_clean_norm = normalize_s2(self.s2_raw)   # [6,256,256] float [0,1]
        self.s1_norm       = normalize_s1(self.s1_raw)   # [2,256,256] z-score

        self.clean_rgb_np  = to_rgb_np(self.s2_clean_norm)   # uint8 para mostrar
        self.sar_gray_np   = to_sar_np(self.s1_raw, band=0)  # uint8

        self.mask      = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        self.drawing   = False
        self.brush     = 20
        self.preds     = None   # dict name -> tensor [6,256,256]
        self.order     = None   # lista aleatoria de nombres

        root.title("MOS — Evaluación de modelos")
        self._build_ui()
        self._show_phase1()

    # ── UI ────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.frame_top    = tk.Frame(self.root); self.frame_top.pack(fill="x", padx=10, pady=6)
        self.label_info   = tk.Label(self.frame_top, text="", font=("Arial", 12)); self.label_info.pack()

        self.frame_canvas = tk.Frame(self.root); self.frame_canvas.pack()
        self.frame_bottom = tk.Frame(self.root); self.frame_bottom.pack(pady=8)

    def _clear_canvas(self):
        for w in self.frame_canvas.winfo_children(): w.destroy()
        for w in self.frame_bottom.winfo_children(): w.destroy()

    # ── FASE 1: dibujar máscara ───────────────────────────────────────

    def _show_phase1(self):
        self._clear_canvas()
        self.mask[:] = 0
        self.label_info.config(text="Dibujá nubes sobre la imagen (pincel blanco). Luego presioná 'Generar predicciones'.")

        self.canvas = tk.Canvas(self.frame_canvas, width=IMG_SIZE, height=IMG_SIZE, cursor="circle")
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>",   self._start_draw)
        self.canvas.bind("<B1-Motion>",       self._draw)
        self.canvas.bind("<ButtonRelease-1>", self._stop_draw)

        self._refresh_canvas()

        ctrl = tk.Frame(self.frame_bottom); ctrl.pack()
        tk.Label(ctrl, text="Tamaño pincel:").pack(side="left")
        self.brush_var = tk.IntVar(value=self.brush)
        tk.Scale(ctrl, from_=5, to=60, orient="horizontal",
                 variable=self.brush_var, command=lambda v: setattr(self, "brush", int(v))
                 ).pack(side="left")
        tk.Button(ctrl, text="Limpiar",              command=self._clear_mask,     width=12).pack(side="left", padx=4)
        tk.Button(ctrl, text="Generar predicciones", command=self._run_predictions, width=20,
                  bg="#4a90d9", fg="white").pack(side="left", padx=4)

    def _refresh_canvas(self):
        display = self.clean_rgb_np.copy()
        display[self.mask == 255] = 255
        img = Image.fromarray(display)
        self._tk_img = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor="nw", image=self._tk_img)

    def _start_draw(self, e): self.drawing = True;  self._paint(e)
    def _stop_draw(self, e):  self.drawing = False
    def _draw(self, e):
        if self.drawing: self._paint(e)

    def _paint(self, e):
        r = self.brush // 2
        x, y = e.x, e.y
        y0, y1 = max(0, y-r), min(IMG_SIZE, y+r)
        x0, x1 = max(0, x-r), min(IMG_SIZE, x+r)
        self.mask[y0:y1, x0:x1] = 255
        self._refresh_canvas()

    def _clear_mask(self):
        self.mask[:] = 0
        self._refresh_canvas()

    # ── FASE 2: inferencia ────────────────────────────────────────────

    def _run_predictions(self):
        if self.mask.sum() == 0:
            tk.messagebox.showwarning("Sin máscara", "Dibujá al menos una nube antes de continuar.")
            return

        self.label_info.config(text="Generando predicciones...")
        self.root.update()

        # Aplicar máscara sobre S2 normalizada
        cloudy_norm = self.s2_clean_norm.copy()
        cloudy_norm[:, self.mask == 255] = 1.0   # blanco en todas las bandas

        cloudy_t = torch.from_numpy(cloudy_norm)
        s1_t     = torch.from_numpy(self.s1_norm)
        clear_t  = torch.from_numpy(self.s2_clean_norm)

        self.preds = {}
        with torch.no_grad():
            for name, (model, sar_mode, mtype) in self.models.items():
                self.preds[name] = run_inference(model, mtype, sar_mode, cloudy_t, s1_t, self.device)

        # Calcular PSNR contra imagen limpia original
        self.psnr_values = {}
        for name, pred in self.preds.items():
            self.psnr_values[name] = psnr(pred.unsqueeze(0), clear_t.unsqueeze(0))

        self.cloudy_norm = cloudy_norm
        self.order = list(self.preds.keys())
        random.shuffle(self.order)

        self._show_phase2()

    # ── FASE 3: mostrar y votar ───────────────────────────────────────

    def _show_phase2(self):
        self._clear_canvas()
        self.label_info.config(text="¿Cuál predicción se ve mejor? Hacé clic en ella.")

        col_labels = ["SAR", "Nublada"] + [f"Pred {i+1}" for i in range(len(self.order))] + ["Original"]
        cols_data  = (
            [("sar",     self.sar_gray_np),
             ("cloudy",  to_rgb_np(self.cloudy_norm))]
            + [("pred",  to_rgb_np(self.preds[n].numpy())) for n in self.order]
            + [("clean", self.clean_rgb_np)]
        )

        self.pred_buttons = []
        for col_idx, (label, (kind, arr)) in enumerate(zip(col_labels, cols_data)):
            frame = tk.Frame(self.frame_canvas, bd=2, relief="flat")
            frame.grid(row=0, column=col_idx, padx=3)

            tk.Label(frame, text=label, font=("Arial", 10, "bold")).pack()

            img = Image.fromarray(arr if arr.ndim == 3 else np.stack([arr]*3, axis=-1))
            tk_img = ImageTk.PhotoImage(img)
            lbl = tk.Label(frame, image=tk_img); lbl.image = tk_img; lbl.pack()

            if kind == "pred":
                pred_name = self.order[col_idx - 2]
                lbl.config(cursor="hand2", relief="flat", bd=3)
                lbl.bind("<Button-1>", lambda e, n=pred_name, f=frame: self._vote(n, f))
                self.pred_buttons.append((pred_name, frame))

        tk.Button(self.frame_bottom, text="← Volver a dibujar", command=self._show_phase1, width=18).pack(side="left", padx=6)

    def _vote(self, chosen_name, chosen_frame):
        # Resaltar selección
        for _, f in self.pred_buttons:
            f.config(bd=2, relief="flat", bg=self.root.cget("bg"))
        chosen_frame.config(bd=4, relief="solid", bg="#4a90d9")

        # Guardar
        self.results["votes"][chosen_name] = self.results["votes"].get(chosen_name, 0) + 1
        session = {
            "chosen": chosen_name,
            "order":  self.order,
            "psnr":   {n: round(v, 4) for n, v in self.psnr_values.items()},
        }
        self.results["sessions"].append(session)
        save_results(self.results)

        self._show_phase3(chosen_name)

    # ── FASE 4: revelar y histograma ──────────────────────────────────

    def _show_phase3(self, chosen_name):
        self._clear_canvas()
        self.label_info.config(text=f"Elegiste la predicción de: {chosen_name}  |  PSNR: {self.psnr_values[chosen_name]:.2f} dB")

        # Mostrar nombres y PSNR de todos
        info_frame = tk.Frame(self.frame_canvas); info_frame.pack(pady=4)
        for i, name in enumerate(self.order):
            psnr_val = self.psnr_values[name]
            marker = "  ◀ tu elección" if name == chosen_name else ""
            tk.Label(info_frame,
                     text=f"Pred {i+1} → {name}  |  PSNR: {psnr_val:.2f} dB{marker}",
                     font=("Arial", 11),
                     fg="#4a90d9" if name == chosen_name else "black"
                     ).pack(anchor="w")

        # Histograma
        hist_frame = tk.Frame(self.frame_canvas); hist_frame.pack(pady=10)
        votes      = self.results["votes"]
        names      = list(votes.keys())
        counts     = [votes[n] for n in names]
        total      = sum(counts)

        bar_w, bar_max_h, pad = 80, 150, 10
        canvas_w = len(names) * (bar_w + pad) + pad
        canvas_h = bar_max_h + 60

        c = tk.Canvas(hist_frame, width=canvas_w, height=canvas_h, bg="white")
        c.pack()

        max_count = max(counts) if counts else 1
        for i, (name, count) in enumerate(zip(names, counts)):
            x0 = pad + i * (bar_w + pad)
            x1 = x0 + bar_w
            h  = int(count / max_count * bar_max_h)
            y0 = bar_max_h - h + 10
            y1 = bar_max_h + 10
            color = "#4a90d9" if name == chosen_name else "#aac8e8"
            c.create_rectangle(x0, y0, x1, y1, fill=color, outline="")
            c.create_text((x0+x1)//2, y1 + 12, text=name.replace(" ", "\n"), font=("Arial", 7), anchor="n")
            c.create_text((x0+x1)//2, y0 - 4,  text=f"{count}/{total}",       font=("Arial", 9, "bold"), anchor="s")

        tk.Button(self.frame_bottom, text="Nueva evaluación", command=self._show_phase1,
                  width=20, bg="#4a90d9", fg="white").pack(pady=6)


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    models = load_models(device)

    root = tk.Tk()
    app  = MOSApp(root, models, device)
    root.mainloop()