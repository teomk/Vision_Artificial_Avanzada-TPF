# La idea
# El modelo DB-CR predice la imagen limpia pred a partir de la imagen nubosa cloudy. En zonas con nube, pred debería diferir mucho de cloudy (el modelo tuvo que "inventar" información). En zonas sin nube, pred debería parecerse a cloudy (no había nada que reconstruir).
# Esto nos da una señal gratuita: el residual |pred - cloudy| como proxy de dónde había nube. En vez de usar una máscara binaria externa (que introduce errores de segmentación), usamos este residual para fusionar suavemente:

# Residual alto → zona nubosa → confiar en pred
# Residual bajo → zona limpia → confiar en cloudy original

# La fusión es suave vía una sigmoide sobre el residual, con dos hiperparámetros: threshold (punto medio de la transición) y sharpness (qué tan abrupta es).
# El código
# pythonimport torch
import torch.nn.functional as F


def adaptive_fusion(pred, cloudy, sharpness=20.0, threshold=0.05):
    """
    Fusión adaptativa post-proceso para DB-CR.

    En zonas donde pred difiere mucho de cloudy (residual alto)
    → probablemente había nube → usa pred.
    En zonas donde pred ≈ cloudy (residual bajo)
    → probablemente estaba limpio → usa cloudy original.

    La transición es suave via sigmoide — no hay bordes abruptos
    ni necesidad de máscara de nube externa.

    Args:
        pred      [B, C, H, W]: predicción del modelo
        cloudy    [B, C, H, W]: imagen nubosa original
        sharpness (float):      pendiente de la sigmoide.
                                Mayor → transición más abrupta.
        threshold (float):      residual en el punto medio de la transición.
                                Calibrar con calibrate_threshold().

    Returns:
        fused   [B, C, H, W]: imagen fusionada
        weight  [B, 1, H, W]: mapa de pesos (0=cloudy, 1=pred)
    """
    # Residual promediado sobre canales → [B, 1, H, W]
    residual = (pred - cloudy).abs().mean(dim=1, keepdim=True)

    # Peso suave en [0, 1]
    weight = torch.sigmoid(sharpness * (residual - threshold))

    # Fusión convexa
    fused = weight * pred + (1 - weight) * cloudy

    return fused, weight


def calibrate_threshold(model, loader, device, sar_mode, T=1000, steps=10,
                        sigmoid_k=10.0, n_batches=50):
    """
    Calcula el threshold óptimo como promedio entre el residual medio
    en zonas sin nube y en zonas con nube, sobre una muestra del dataset.

    Requiere que el loader incluya máscaras (include_mask=True).

    Args:
        model:      DBCRSimple o DBCR
        loader:     DataLoader con include_mask=True
        device:     torch.device
        sar_mode:   "None" | "Concat" | "ControlNet"
        n_batches:  cuántos batches usar para calibrar

    Returns:
        threshold (float): valor recomendado para usar en adaptive_fusion
    """
    from dbcr_simple_utils import inference, make_bridge_sample
    from dataset_utils import unpack_batch

    model.eval()
    residuals_clean  = []
    residuals_cloudy = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break

            # batch con máscara: (s1, cloudy, mask, clean) o (cloudy, mask, clean)
            if sar_mode != "None":
                s1, cloudy, mask, clean = batch
                s1     = s1.to(device)
                cloudy = cloudy.to(device)
                mask   = mask.to(device)
                clean  = clean.to(device)
                condition = torch.cat([cloudy, s1], dim=1) if sar_mode == "Concat" else cloudy
                sar       = None if sar_mode == "Concat" else s1
            else:
                cloudy, mask, clean = batch
                cloudy = cloudy.to(device)
                mask   = mask.to(device)
                clean  = clean.to(device)
                condition = cloudy
                sar       = None

            pred = inference(
                model, cloudy, condition, device,
                T=T, steps=steps, sar=sar, sigmoid_k=sigmoid_k
            ).clamp(0, 1)

            residual = (pred - cloudy).abs().mean(dim=1, keepdim=True)  # [B, 1, H, W]

            # mask: 1 = nube, 0 = limpio
            mask_bool = mask.bool()

            residuals_clean.append(residual[~mask_bool].mean().item())
            residuals_cloudy.append(residual[mask_bool].mean().item())

    mean_clean  = sum(residuals_clean)  / len(residuals_clean)
    mean_cloudy = sum(residuals_cloudy) / len(residuals_cloudy)
    threshold   = (mean_clean + mean_cloudy) / 2.0

    print(f"Residual medio zonas limpias : {mean_clean:.4f}")
    print(f"Residual medio zonas nubosas : {mean_cloudy:.4f}")
    print(f"Threshold recomendado        : {threshold:.4f}")

    return threshold


def evaluate_with_fusion(model, loader, device, sar_mode, T=1000, steps=10,
                          sigmoid_k=10.0, sharpness=20.0, threshold=0.05):
    """
    Eval completo con adaptive_fusion aplicado post-inferencia.
    Compara métricas con y sin fusión para ver si mejora.

    Requiere loader con include_mask=False (las métricas se calculan
    sobre la imagen completa, no solo zonas nubosas).
    """
    from dbcr_simple_utils import inference
    from dataset_utils import unpack_batch
    from metrics import mae, psnr, ssim, sam

    model.eval()

    metrics_raw    = {"mae": 0, "psnr": 0, "ssim": 0, "sam": 0}
    metrics_fused  = {"mae": 0, "psnr": 0, "ssim": 0, "sam": 0}
    n = 0

    with torch.no_grad():
        for batch in loader:
            s2_cloudy, s2_clean, condition, sar = unpack_batch(batch, sar_mode, device)

            pred = inference(
                model, s2_cloudy, condition, device,
                T=T, steps=steps, sar=sar, sigmoid_k=sigmoid_k
            ).clamp(0, 1)

            fused, _ = adaptive_fusion(pred, s2_cloudy,
                                        sharpness=sharpness,
                                        threshold=threshold)
            fused = fused.clamp(0, 1)

            metrics_raw["mae"]  += mae(pred,  s2_clean)
            metrics_raw["psnr"] += psnr(pred, s2_clean)
            metrics_raw["ssim"] += ssim(pred, s2_clean)
            metrics_raw["sam"]  += sam(pred,  s2_clean)

            metrics_fused["mae"]  += mae(fused,  s2_clean)
            metrics_fused["psnr"] += psnr(fused, s2_clean)
            metrics_fused["ssim"] += ssim(fused, s2_clean)
            metrics_fused["sam"]  += sam(fused,  s2_clean)

            n += 1

    print(f"\n{'='*50}")
    print(f"{'Métrica':<10} {'Sin fusión':>12} {'Con fusión':>12} {'Delta':>10}")
    print(f"{'-'*50}")
    for k in ["mae", "psnr", "ssim", "sam"]:
        raw   = metrics_raw[k]   / n
        fused_val = metrics_fused[k] / n
        delta = fused_val - raw
        arrow = "↑" if (k in ["psnr", "ssim"] and delta > 0) or (k in ["mae", "sam"] and delta < 0) else "↓"
        print(f"{k.upper():<10} {raw:>12.4f} {fused_val:>12.4f} {delta:>+10.4f} {arrow}")
    print(f"{'='*50}\n")

    return metrics_raw, metrics_fused
# Cómo usarlo
# python# 1. Calibrar threshold con validation set (necesita máscaras)
# threshold = calibrate_threshold(model, val_loader_with_mask, device, sar_mode="None")

# # 2. Evaluar con y sin fusión
# metrics_raw, metrics_fused = evaluate_with_fusion(
#     model, test_loader, device, sar_mode="None",
#     sharpness=20.0, threshold=threshold
# )

# # 3. En visualización, aplicar post-proceso
# pred = inference(...)
# fused, weight_map = adaptive_fusion(pred, cloudy, sharpness=20.0, threshold=threshold)
# El weight_map también es útil para visualizar — podés mostrarlo como un cuarto panel junto a cloudy/pred/clear para ver dónde el modelo decidió confiar en su propia predicción vs en la imagen original.