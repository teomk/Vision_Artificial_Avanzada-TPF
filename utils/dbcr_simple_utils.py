import torch

def sigmoid_scheduler(T, sigmoid_k, t, device):
    tau   = torch.clamp(t.float() / T, 0.0, 1.0)
    s     = torch.sigmoid((tau - 0.5) * sigmoid_k)
    s_min = torch.sigmoid(torch.tensor(-0.5 * sigmoid_k, device=device))
    s_max = torch.sigmoid(torch.tensor( 0.5 * sigmoid_k, device=device))
    alpha = torch.clamp((s - s_min) / (s_max - s_min), 0.0, 1.0)
    return alpha[:, None, None, None]

def make_bridge_sample(s2_clean, s2_cloudy, t, T, sigmoid_k, device):
    alpha_t = sigmoid_scheduler(T, sigmoid_k, t, device)
    return (1.0 - alpha_t) * s2_clean + alpha_t * s2_cloudy

def inference(model, cloudy_b, condition, device, T=1000, steps=10, sar=None, sigmoid_k=10.0):
    """
    cloudy_b:  [1, 6, H, W] — solo S2, usado para el bridge (x_t)
    condition: [1, 6, H, W] o [1, 8, H, W] — lo que recibe el modelo como s2_cloudy
               (6ch para No-SAR, 8ch para SAR-concat)
    """
    x_t = cloudy_b.clone()
    timesteps = torch.linspace(T, 1, steps).long().to(device)
    with torch.no_grad():
        for t_val in timesteps:
            B = x_t.shape[0]
            t = t_val.repeat(B).to(device)
            pred_clean = model(x_t=x_t, t=t, s2_cloudy=condition, sar=sar)
            prev_t = (t_val - (T // steps)).clamp(min=1)
            prev_t = prev_t.repeat(B).to(device)
            x_t = make_bridge_sample(s2_clean=pred_clean, s2_cloudy=cloudy_b, t=prev_t, T=T, sigmoid_k=sigmoid_k, device=device).clamp(0, 1)
    return pred_clean