import torch
from diffusers import DDPMScheduler
from tqdm.auto import tqdm

def make_sigmoid_alpha_bar_schedule(T, sigmoid_k=25.0, alpha_min=1e-4):
    """
    Crea alpha_bar[t] sigmoidal para t=0,...,T-1.

    alpha_bar[0]  = 1.0       -> sin ruido
    alpha_bar[-1] = alpha_min -> casi ruido puro
    """

    T = int(T)
    sigmoid_k = float(sigmoid_k)
    alpha_min = float(alpha_min)

    tau = torch.linspace(0.0, 1.0, T)

    s = torch.sigmoid((tau - 0.5) * sigmoid_k)

    s_min = torch.sigmoid(torch.tensor(-0.5 * sigmoid_k))
    s_max = torch.sigmoid(torch.tensor( 0.5 * sigmoid_k))

    progress = (s - s_min) / (s_max - s_min)
    progress = torch.clamp(progress, 0.0, 1.0)

    alpha_bar = (1.0 - progress) * (1.0 - alpha_min) + alpha_min

    alpha_bar[0] = 1.0
    alpha_bar[-1] = alpha_min

    return alpha_bar


def alpha_bar_to_betas(alpha_bar, max_beta=0.999):
    """
    Convierte alpha_bar acumulado en betas para DDPM.

    alpha_bar_t = prod(1 - beta_i)
    beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}
    """

    betas = torch.zeros_like(alpha_bar)

    betas[0] = 1.0 - alpha_bar[0]
    betas[1:] = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])

    betas[0] = 1e-4

    betas = torch.clamp(betas, min=1e-6, max=max_beta)

    return betas


def build_sigmoid_ddpm_scheduler(T, sigmoid_k=25.0, alpha_min=1e-4):
    """
    Scheduler DDPM idéntico al del train.
    """

    alpha_bar = make_sigmoid_alpha_bar_schedule(
        T=T,
        sigmoid_k=sigmoid_k,
        alpha_min=alpha_min
    )
    print(f"alpha_bar[0]   = {alpha_bar[0]:.4f}")    # debería ser ~1.0
    print(f"alpha_bar[500] = {alpha_bar[500]:.4f}")  # debería ser ~0.5
    print(f"alpha_bar[999] = {alpha_bar[999]:.4f}")  # debería ser ~alpha_min

    betas = alpha_bar_to_betas(alpha_bar)

    scheduler = DDPMScheduler(
        num_train_timesteps=int(T),
        trained_betas=betas.cpu().numpy(),
        prediction_type="epsilon",
        clip_sample=True,
    )

    return scheduler


def inference(model, condition, device, scheduler, steps=50, sar=None):
    """
    Inferencia DDPM reversa: x_T (ruido puro) → x_0 (imagen limpia).

    condition: [B, 6 u 8, H, W]
    """
    B, C_cond, H, W = condition.shape
    image_channels  = model.out[-1].out_channels

    x_t = torch.randn(B, image_channels, H, W, device=device)

    scheduler.set_timesteps(steps)

    with torch.no_grad():
        for t_val in tqdm(scheduler.timesteps, desc="DDPM", unit="step", leave=False):
            t          = t_val.repeat(B).to(device)
            noise_pred = model(x_t=x_t, t=t, s2_cloudy=condition, sar=sar)
            x_t        = scheduler.step(noise_pred, t_val, x_t).prev_sample

    return x_t.clamp(0, 1)
