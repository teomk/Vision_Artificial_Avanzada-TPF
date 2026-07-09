import tkinter as tk
from tkinter import ttk, messagebox
import json
import random
import time
import numpy as np
import torch
import sys
from pathlib import Path
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
UTILS_DIR = ROOT / "utils"
DATA_DIR = ROOT / "dataset"

sys.path.append(str(DATA_DIR))
sys.path.append(str(MODELS_DIR))
sys.path.append(str(UTILS_DIR))

from dbcr_simple import DBCRSimple
from dbcr_complex import DBCR
from hf_utils import download_model
from dbcr_utils import inference
from dataset_utils import unpack_batch
from evaluate_utils import psnr

S2_PATH = ROOT / "visualize" / "s2_6bands.npy"
S1_PATH = ROOT / "visualize" / "s1_2bands.npy"
RESULTS_PATH = ROOT / "visualize" / "mos_results.json"
CACHE_DIR = ROOT / "visualize" / "inference_cache"
CLOUDY_IMAGES_DIR = ROOT / "visualize"

CACHE_DIR.mkdir(parents=True, exist_ok=True)

CLOUDY_IMAGES = {
    "Cloudy 01": CLOUDY_IMAGES_DIR / "s2_6bands_cloudy_01.npy",
    "Cloudy 02": CLOUDY_IMAGES_DIR / "s2_6bands_cloudy_02.npy",
    "Cloudy 03": CLOUDY_IMAGES_DIR / "s2_6bands_cloudy_03.npy",
}

REPO_ID = "LucioLuque/lama"
T = 1000
STEPS = 10
SIGMOID_K = 10.0

S1_MEAN = np.array([-8.999908447265625, -14.78221321105957], dtype=np.float32)
S1_STD  = np.array([2.413282871246338,  2.3029115200042725], dtype=np.float32)

IMG_SIZE = 256

def normalize_s2(s2_raw):
    return np.clip(s2_raw / 10000.0, 0, 1).astype(np.float32)

def normalize_s1(s1_raw):
    mean = S1_MEAN[:, None, None]
    std = S1_STD[:, None, None]
    return ((s1_raw - mean) / (std + 1e-6)).astype(np.float32)

def to_rgb_np(arr_chw, bands=(2, 1, 0)):
    if arr_chw.ndim == 3 and arr_chw.shape[0] >= 3:
        arr = arr_chw[list(bands)].transpose(1, 2, 0)
    else:
        arr = arr_chw
    p2, p98 = np.percentile(arr, 2), np.percentile(arr, 98)
    arr = np.clip((arr - p2) / (p98 - p2 + 1e-8), 0, 1)
    return (arr * 255).astype(np.uint8)

def to_sar_np(s1_raw, band=0):
    img = s1_raw[band]
    p2, p98 = np.percentile(img, 2), np.percentile(img, 98)
    img = np.clip((img - p2) / (p98 - p2 + 1e-8), 0, 1)
    return (img * 255).astype(np.uint8)

def format_seconds(seconds):
    seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"

def load_models(device):
    print("Cargando modelos...")

    def load_simple(filename, sar_mode):
        condition_channels = 8 if sar_mode == "Concat" else 6
        ckpt = download_model(repo_id=REPO_ID, filename=filename, map_location=device)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model = DBCRSimple(image_channels=6, condition_channels=condition_channels,
                           base_channels=64, time_dim=128, control_net=(sar_mode == "ControlNet"))
        model.load_state_dict(state, strict=False)
        return model.float().to(device).eval()

    def load_complex(filename):
        ckpt = download_model(repo_id=REPO_ID, filename=filename, map_location=device)
        model = DBCR(image_channels=6, condition_channels=6, sar_channels=2,
                     base_channels=64, time_dim=128, num_heads=1,
                     window_size_sf0=8, window_size_not_sf0=None,
                     use_checkpoint=True, include_encoder_4=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        return model.float().to(device).eval()

    return {
        "DBCR-S": (load_simple("dbcr_no_sar_naf_v2.pth", "None"),      "None",       "simple"),
        "DBCR-SC (sin TL)": (load_simple("dbcr_concat_v2.pth",     "Concat"),    "Concat",     "simple"),
        "DBCR": (load_complex("dbcr_complex_v7.pth"),                 "ControlNet", "complex"),
    }

def get_cache_path(image_name):
    cache_subdir = CACHE_DIR / image_name.replace(" ", "_")
    cache_subdir.mkdir(parents=True, exist_ok=True)
    return cache_subdir

def save_predictions_cache(image_name, preds, psnr_values):
    cache_path = get_cache_path(image_name)
    
    for model_name, pred_tensor in preds.items():
        pred_file = cache_path / f"{model_name.replace(' ', '_')}_pred.pt"
        torch.save(pred_tensor, pred_file)
    
    psnr_file = cache_path / "psnr_values.pt"
    torch.save(psnr_values, psnr_file)
    
def load_predictions_cache(image_name):
    cache_path = get_cache_path(image_name)
    
    psnr_file = cache_path / "psnr_values.pt"
    if not psnr_file.exists():
        return None, None
    
    preds = {}
    model_files = list(cache_path.glob("*_pred.pt"))
    
    for pred_file in model_files:
        model_name = pred_file.stem.replace("_pred", "").replace("_", " ")
        preds[model_name] = torch.load(pred_file, map_location="cpu")
    
    psnr_values = torch.load(psnr_file, map_location="cpu")
    
    return preds, psnr_values

def run_inference(model, model_type, sar_mode, cloudy_t, s1_t, device):
    cloudy_b = cloudy_t.unsqueeze(0).float().to(device)
    if model_type == "simple":
        if sar_mode == "None":
            condition, sar = cloudy_b, None
        elif sar_mode == "Concat":
            s1_b = s1_t.unsqueeze(0).float().to(device)
            condition = torch.cat([cloudy_b, s1_b], dim=1)
            sar = None
        else:
            s1_b = s1_t.unsqueeze(0).float().to(device)
            condition = cloudy_b
            sar = s1_b
        pred = inference(model, cloudy_b, condition, device, T=T, steps=STEPS, sar=sar, sigmoid_k=SIGMOID_K, show_progress=False)
    else:
        fake_batch = (s1_t.unsqueeze(0).float(), cloudy_b, cloudy_b)
        s2_cloudy, _, cond, sar = unpack_batch(fake_batch, sar_mode, device)
        pred = inference(model, s2_cloudy, cond, device, T=T, steps=STEPS, sar=sar, sigmoid_k=SIGMOID_K)
    return pred.squeeze(0).clamp(0, 1).cpu()

def load_results():
    if RESULTS_PATH.exists():
        try:
            with open(RESULTS_PATH) as f:
                content = f.read()
                if content.strip():
                    results = json.loads(content)
                else:
                    results = {}
        except (json.JSONDecodeError, IOError):
            results = {}
    else:
        results = {}

    results.setdefault("votes", {})
    results.setdefault("rankings", {})
    results.setdefault("points", {})
    results.setdefault("sessions", [])

    return results

def save_results(results):
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

class MOSApp:
    def __init__(self, root, models, device):
        self.root = root
        self.models = models
        self.device = device
        self.results = load_results()

        self.s2_raw = np.load(S2_PATH)
        self.s1_raw = np.load(S1_PATH)
        self.s2_clean_norm = normalize_s2(self.s2_raw)
        self.s1_norm = normalize_s1(self.s1_raw)
        self.clean_rgb_np = to_rgb_np(self.s2_clean_norm)
        self.sar_gray_np = to_sar_np(self.s1_raw, band=0)

        self.selected_image_name = None
        self.cloudy_norm = None
        self.preds = None
        self.order = None
        self.psnr_values = None

        self.current_ranking = []
        self.pred_frames = {}
        self.pred_labels = {}
                
        root.title("MOS — Evaluación de modelos")
        self._build_ui()
        self._show_phase1()

    def _build_ui(self):
        self.frame_top = tk.Frame(self.root)
        self.frame_top.pack(fill="x", padx=10, pady=6)
        self.label_info = tk.Label(self.frame_top, text="", font=("Arial", 12))
        self.label_info.pack()

        self.scroll_container = tk.Frame(self.root)
        self.scroll_container.pack(fill="both", expand=True)

        self.canvas_scroll = tk.Canvas(self.scroll_container)
        self.scrollbar_x = tk.Scrollbar(self.scroll_container, orient="horizontal", command=self.canvas_scroll.xview)
        self.scrollbar_y = tk.Scrollbar(self.scroll_container, orient="vertical", command=self.canvas_scroll.yview)
        self.canvas_scroll.configure(xscrollcommand=self.scrollbar_x.set, yscrollcommand=self.scrollbar_y.set)

        self.scrollbar_x.pack(side="bottom", fill="x")
        self.scrollbar_y.pack(side="right", fill="y")
        self.canvas_scroll.pack(side="left", fill="both", expand=True)

        self.frame_canvas = tk.Frame(self.canvas_scroll)
        self.canvas_scroll.create_window((0, 0), window=self.frame_canvas, anchor="nw")
        self.frame_canvas.bind("<Configure>", lambda e: self.canvas_scroll.configure(scrollregion=self.canvas_scroll.bbox("all")))

        self.frame_bottom = tk.Frame(self.root)
        self.frame_bottom.pack(pady=8)

    def _clear_canvas(self):
        for w in self.frame_canvas.winfo_children(): w.destroy()
        for w in self.frame_bottom.winfo_children(): w.destroy()

    def _show_phase1(self):
        self._clear_canvas()
        self.label_info.config(text="Selecciona una imagen nublada. Luego presioná 'Generar/Cargar predicciones'.")

        left = tk.Frame(self.frame_canvas)
        left.grid(row=0, column=0, padx=10, pady=10, sticky="n")

        tk.Label(left, text="Selecciona imagen:", font=("Arial", 11, "bold")).pack(anchor="w", pady=(0, 10))

        self.selected_button = None
        
        for image_name in CLOUDY_IMAGES.keys():
            btn = tk.Button(
                left, 
                text=image_name, 
                command=lambda name=image_name: self._select_image(name),
                width=18, 
                bg="#95a5a6", 
                fg="white", 
                font=("Arial", 10),
                height=2
            )
            btn.pack(pady=6)
            btn.image_name = image_name

        tk.Button(
            left, 
            text="Generar/Cargar →", 
            command=self._run_predictions,
            width=20, 
            bg="#27ae60", 
            fg="white", 
            font=("Arial", 10),
            height=2
        ).pack(pady=16)

        right = tk.Frame(self.frame_canvas)
        right.grid(row=0, column=1, padx=10, pady=10)

        tk.Label(right, text="Preview", font=("Arial", 10, "bold")).pack()
        self.preview_label = tk.Label(right, bg="#f0f0f0")
        self.preview_label.pack(pady=4)

    def _select_image(self, image_name):
        self.selected_image_name = image_name        
        for widget in self.frame_canvas.winfo_children():
            if isinstance(widget, tk.Frame):
                for btn in widget.winfo_children():
                    if isinstance(btn, tk.Button) and hasattr(btn, 'image_name'):
                        if btn.image_name == image_name:
                            btn.config(bg="#27ae60")
                        else:
                            btn.config(bg="#95a5a6")
        
        image_path = CLOUDY_IMAGES[image_name]
        cloudy_raw = np.load(image_path)
        cloudy_norm = normalize_s2(cloudy_raw)
        cloudy_rgb = to_rgb_np(cloudy_norm)
        
        img = Image.fromarray(cloudy_rgb)
        img = img.resize((512, 512), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(img)
        self.preview_label.config(image=tk_img)
        self.preview_label.image = tk_img

    def _run_predictions(self):
        if self.selected_image_name is None:
            messagebox.showwarning("Sin imagen", "Selecciona una imagen primero.")
            return

        self.label_info.config(text="Verificando caché e iniciando predicciones...")
        self.root.update()

        image_path = CLOUDY_IMAGES[self.selected_image_name]
        cloudy_raw = np.load(image_path)
        self.cloudy_norm = normalize_s2(cloudy_raw)
        
        cached_preds, cached_psnr = load_predictions_cache(self.selected_image_name)
        
        if cached_preds is not None and cached_psnr is not None:
            self.preds = cached_preds
            self.psnr_values = cached_psnr
            self.label_info.config(text="✓ Predicciones cargadas desde caché")
            self.root.update_idletasks()
        else:
            cloudy_t = torch.from_numpy(self.cloudy_norm)
            s1_t = torch.from_numpy(self.s1_norm)
            clear_t = torch.from_numpy(self.s2_clean_norm)

            self.preds = {}
            model_items = list(self.models.items())
            total_models = len(model_items)
            start_time = time.perf_counter()

            with torch.no_grad():
                for index, (name, (model, sar_mode, mtype)) in enumerate(model_items, start=1):
                    self.label_info.config(text=f"Generando predicciones... {index}/{total_models}")
                    self.root.update_idletasks()

                    self.preds[name] = run_inference(model, mtype, sar_mode, cloudy_t, s1_t, self.device)

                    elapsed = time.perf_counter() - start_time
                    remaining = (elapsed / index) * (total_models - index) if index else 0.0
                    self.label_info.config(
                        text=(
                            f"Generando predicciones... {index}/{total_models} | "
                            f"transcurrido {format_seconds(elapsed)} | "
                            f"restante aprox. {format_seconds(remaining)}"
                        )
                    )
                    self.root.update_idletasks()

            self.psnr_values = {
                name: psnr(pred.unsqueeze(0), clear_t.unsqueeze(0))
                for name, pred in self.preds.items()
            }
            
            save_predictions_cache(self.selected_image_name, self.preds, self.psnr_values)

        self.order = list(self.preds.keys())
        random.shuffle(self.order)
        self._show_phase2()

    def _show_phase2(self):
        self._clear_canvas()
        self.current_ranking = []
        self.pred_frames = {}
        self.pred_labels = {}

        self.label_info.config(text="Rankeá las predicciones: clic 1 = mejor, clic 2 = segunda, clic 3 = tercera.")

        col_labels = ["SAR", "Nublada"] + [f"Pred {i+1}" for i in range(len(self.order))] + ["Original"]
        cols_data = (
            [("sar", self.sar_gray_np),
            ("cloudy", to_rgb_np(self.cloudy_norm))]
            + [("pred", to_rgb_np(self.preds[n].numpy())) for n in self.order]
            + [("clean", self.clean_rgb_np)]
        )

        self.pred_buttons = []

        for col_idx, (label, (kind, arr)) in enumerate(zip(col_labels, cols_data)):
            frame = tk.Frame(self.frame_canvas, bd=2, relief="flat")
            frame.grid(row=0, column=col_idx, padx=3, pady=4)

            title_lbl = tk.Label(frame, text=label, font=("Arial", 10, "bold"))
            title_lbl.pack()

            display = arr if arr.ndim == 3 else np.stack([arr] * 3, axis=-1)
            img = Image.fromarray(display)
            tk_img = ImageTk.PhotoImage(img)

            lbl = tk.Label(frame, image=tk_img)
            lbl.image = tk_img
            lbl.pack()

            rank_lbl = tk.Label(frame, text="", font=("Arial", 12, "bold"), fg="#4a90d9")
            rank_lbl.pack(pady=4)

            if kind == "pred":
                pred_name = self.order[col_idx - 2]

                lbl.config(cursor="hand2", bd=3, relief="flat")
                lbl.bind("<Button-1>", lambda e, n=pred_name: self._select_rank(n))

                frame.bind("<Button-1>", lambda e, n=pred_name: self._select_rank(n))

                self.pred_buttons.append((pred_name, frame))
                self.pred_frames[pred_name] = frame
                self.pred_labels[pred_name] = rank_lbl

        tk.Button(
            self.frame_bottom,
            text="← Volver a generar nube",
            command=self._show_phase1,
            width=22
        ).pack(side="left", padx=6)

        tk.Button(
            self.frame_bottom,
            text="Reiniciar ranking",
            command=self._reset_ranking,
            width=18
        ).pack(side="left", padx=6)

    def _select_rank(self, pred_name):
        if pred_name in self.current_ranking:
            messagebox.showinfo(
                "Ya seleccionada",
                "Esa predicción ya fue rankeada. Usá 'Reiniciar ranking' si querés cambiar el orden."
            )
            return

        if len(self.current_ranking) >= len(self.order):
            return

        self.current_ranking.append(pred_name)
        rank = len(self.current_ranking)

        frame = self.pred_frames[pred_name]
        label = self.pred_labels[pred_name]

        colors = {
            1: "#27ae60",  # verde
            2: "#f1c40f",  # amarillo
            3: "#e67e22",  # naranja
        }

        frame.config(
            bd=5,
            relief="solid",
            bg=colors.get(rank, "#4a90d9")
        )

        label.config(
            text=f"Puesto {rank}",
            fg=colors.get(rank, "#4a90d9")
        )

        faltan = len(self.order) - len(self.current_ranking)

        if faltan > 0:
            self.label_info.config(
                text=f"Ranking parcial: elegiste {rank}/{len(self.order)}. Faltan {faltan}."
            )
        else:
            self._confirm_ranking()

    def _confirm_ranking(self):
        ranking = list(self.current_ranking)

        if len(ranking) != len(self.order):
            messagebox.showwarning(
                "Ranking incompleto",
                "Tenés que rankear todas las predicciones."
            )
            return

        best_name = ranking[0]

        self.results["votes"][best_name] = self.results["votes"].get(best_name, 0) + 1

        for name in ranking:
            self.results["rankings"].setdefault(
                name,
                {"rank_1": 0, "rank_2": 0, "rank_3": 0}
            )
            self.results["points"].setdefault(name, 0)

        total_preds = len(ranking)

        for pos, name in enumerate(ranking, start=1):
            rank_key = f"rank_{pos}"
            points = total_preds - pos + 1

            self.results["rankings"][name][rank_key] += 1
            self.results["points"][name] += points

        self.results["sessions"].append({
            "ranking": ranking,
            "order": self.order,
            "psnr": {n: round(float(v), 4) for n, v in self.psnr_values.items()},
        })

        save_results(self.results)
        self._show_phase3(ranking)

    def _reset_ranking(self):
        self.current_ranking = []

        for name, frame in self.pred_frames.items():
            frame.config(
                bd=2,
                relief="flat",
                bg=self.root.cget("bg")
            )

        for name, label in self.pred_labels.items():
            label.config(text="")

        self.label_info.config(
            text="Ranking reiniciado. Clic 1 = mejor, clic 2 = segunda, clic 3 = tercera."
        )

    def _show_phase3(self, ranking):
        self._clear_canvas()

        best_name = ranking[0]

        self.label_info.config(
            text=(
                f"Ranking guardado. "
                f"1°: {ranking[0]} | "
                f"2°: {ranking[1]} | "
                f"3°: {ranking[2]}"
            )
        )

        info_frame = tk.Frame(self.frame_canvas)
        info_frame.pack(pady=8)

        tk.Label(
            info_frame,
            text="Ranking elegido",
            font=("Arial", 12, "bold")
        ).pack(anchor="w", pady=(0, 6))

        for pos, name in enumerate(ranking, start=1):
            tk.Label(
                info_frame,
                text=f"{pos}° → {name}  |  PSNR: {self.psnr_values[name]:.2f} dB",
                font=("Arial", 11),
                fg="#4a90d9" if pos == 1 else "black"
            ).pack(anchor="w")

        tk.Label(
            info_frame,
            text="\nCorrespondencia visual de esta ronda",
            font=("Arial", 12, "bold")
        ).pack(anchor="w", pady=(10, 6))

        for i, name in enumerate(self.order):
            puesto = ranking.index(name) + 1
            tk.Label(
                info_frame,
                text=(
                    f"Pred {i+1} → {name} | "
                    f"Puesto elegido: {puesto}° | "
                    f"PSNR: {self.psnr_values[name]:.2f} dB"
                ),
                font=("Arial", 11)
            ).pack(anchor="w")

        table_frame = tk.Frame(self.frame_canvas)
        table_frame.pack(pady=12)

        tk.Label(
            table_frame,
            text="Resultados acumulados",
            font=("Arial", 12, "bold")
        ).grid(row=0, column=0, columnspan=6, pady=(0, 6))

        headers = ["Modelo", "1°", "2°", "3°", "Puntos", "PSNR ronda"]
        for col, h in enumerate(headers):
            tk.Label(
                table_frame,
                text=h,
                font=("Arial", 10, "bold"),
                borderwidth=1,
                relief="solid",
                padx=8,
                pady=4
            ).grid(row=1, column=col, sticky="nsew")

        all_names = list(self.models.keys())

        all_names = sorted(
            all_names,
            key=lambda n: self.results["points"].get(n, 0),
            reverse=True
        )

        for row, name in enumerate(all_names, start=2):
            rank_stats = self.results["rankings"].get(
                name,
                {"rank_1": 0, "rank_2": 0, "rank_3": 0}
            )

            values = [
                name,
                rank_stats.get("rank_1", 0),
                rank_stats.get("rank_2", 0),
                rank_stats.get("rank_3", 0),
                self.results["points"].get(name, 0),
                f"{self.psnr_values[name]:.2f} dB" if name in self.psnr_values else "-",
            ]

            for col, value in enumerate(values):
                tk.Label(
                    table_frame,
                    text=value,
                    font=("Arial", 10),
                    borderwidth=1,
                    relief="solid",
                    padx=8,
                    pady=4,
                    fg="#4a90d9" if name == best_name else "black"
                ).grid(row=row, column=col, sticky="nsew")

        tk.Button(
            self.frame_bottom,
            text="Nueva evaluación",
            command=self._show_phase1,
            width=20,
            bg="#4a90d9",
            fg="white"
        ).pack(pady=6)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    models = load_models(device)
    root = tk.Tk()
    app = MOSApp(root, models, device)
    root.mainloop()