import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
import math

class GaussianDiffusion(nn.Module):
    def __init__(self, model, image_size, channels=3, timesteps=1000,
                 objective="pred_noise", beta_schedule="linear"):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.channels = channels
        self.num_timesteps = timesteps  # 统一用 num_timesteps
        self.objective = objective

        assert objective in {"pred_noise", "pred_x_start"}, \
            "objective must be either pred_noise or pred_x_start"

        register_buffer = lambda name, val: self.register_buffer(name, val.float())

        betas = get_beta_schedule(beta_schedule, timesteps)  # 传字符串
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], [1, 0], value=1.0)

        register_buffer('alphas', alphas)
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )
        posterior_var = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_std = torch.sqrt(posterior_var.clamp(min=1e-20))
        register_buffer("posterior_std", posterior_std)

        # loss weight: pred_noise 用均匀权重，pred_x_start 用 SNR 加权
        snr = alphas_cumprod / (1 - alphas_cumprod)
        loss_weight = torch.ones_like(snr) if objective == "pred_noise" else snr
        register_buffer("loss_weight", loss_weight)

    def normalize(self, img):
        """[0, 1] -> [-1, 1]"""
        return img * 2 - 1

    def unnormalize(self, img):
        """[-1, 1] -> [0, 1]"""
        return (img + 1) * 0.5

    def q_sample(self, x, t, noise=None):
        """Sample from q(x_t | x_0) according to Eq. (4) of the paper.

        Args:
            x_start: (b, *) tensor. Starting image.
            t: (b,) tensor. Time step.
            noise: (b, *) tensor. Noise from N(0, 1).
        Returns:
            x_t: (b, *) tensor. Noisy image.
        """
        if noise is None:
            noise = torch.randn_like(x)
        
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        return sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise
    
    def predict_start_from_noise(self, x_t, t, noise):
        """Get x_start from x_t and noise according to Eq. (14) of the paper.
        Args:
            x_t: (b, *) tensor. Noisy image.
            t: (b,) tensor. Time step.
            noise: (b, *) tensor. Noise from N(0, 1).
        Returns:
            x_start: (b, *) tensor. Starting image.
        """
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        x_start = (x_t - sqrt_one_minus_alphas_cumprod_t * noise) / sqrt_alphas_cumprod_t
        return x_start
    
    def predict_noise_from_start(self, x_t, t, x_start):
        """Get noise from x_t and x_start according to Eq. (14) of the paper.
        Args:
            x_t: (b, *) tensor. Noisy image.
            t: (b,) tensor. Time step.
            x_start: (b, *) tensor. Starting image.
        Returns:
            pred_noise: (b, *) tensor. Predicted noise.
        """
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        pred_noise = (x_t - sqrt_alphas_cumprod_t * x_start) / sqrt_one_minus_alphas_cumprod_t
        return pred_noise
    
    def q_posterior(self, x_start, x_t, t):
        """Get the posterior q(x_{t-1} | x_t, x_0) according to Eq. (6) and (7) of the paper.
        Args:
            x_start: (b, *) tensor. Predicted start image.
            x_t: (b, *) tensor. Noisy image.
            t: (b,) tensor. Time step.
        Returns:
            posterior_mean: (b, *) tensor. Mean of the posterior.
            posterior_std: (b, *) tensor. Std of the posterior.
        """
        c1 = extract(self.posterior_mean_coef1, t, x_t.shape)
        c2 = extract(self.posterior_mean_coef2, t, x_t.shape)
        posterior_mean = c1 * x_start + c2 * x_t
        posterior_std = extract(self.posterior_std, t, x_t.shape)
        return posterior_mean, posterior_std
    
    @torch.no_grad()
    def p_sample(self, x_t, t: int, model_kwargs={}):
        B, *_ = x_t.shape 
        t = torch.full((B,), t, device=x_t.device, dtype=torch.long)
        model_output = self.model(x_t, t, **model_kwargs)
        
        if self.objective == "pred_noise":
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x_t, t, pred_noise)
        else:
            x_start = model_output
        
        # 4. clamp x_start 到 [-1, 1]（保持稳定）
        x_start = x_start.clamp(-1, 1)
        mean, std = self.q_posterior(x_start, x_t, t)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        x_tm1 = mean + nonzero_mask * std * noise
        
        return x_tm1

    # TODO
    @torch.no_grad()
    def sample(self, batch_size=16, return_all_timesteps=False, model_kwargs={}):
        shape = (batch_size, self.channels, self.image_size, self.image_size)
        img = torch.randn(shape, device=self.betas.device)
        imgs = [img]

        for t in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            img = self.p_sample(img, t, model_kwargs=model_kwargs)
            imgs.append(img)

        res = img if not return_all_timesteps else torch.stack(imgs, dim=1)
        res = self.unnormalize(res)
        return res

    def p_losses(self, x_start, model_kwargs={}):
        b, nts = x_start.shape[0], self.num_timesteps
        t = torch.randint(0, nts, (b,), device=x_start.device).long()  # (b,)
        x_start = self.normalize(x_start)  # (b, *)
        noise = torch.randn_like(x_start)  # (b, *)
        target = noise if self.objective == "pred_noise" else x_start  # (b, *)
        loss_weight = extract(self.loss_weight, t, target.shape)  # (b, *)

        x_t = self.q_sample(x_start, t, noise)
        model_output = self.model(x_t, t, **model_kwargs)
        loss = F.mse_loss(model_output, target, reduction='none')
        loss = (loss * loss_weight).mean()

        return loss
    
def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (
        v_end - v_start
    )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def get_beta_schedule(beta_schedule, timesteps):
    if beta_schedule == "linear":
        beta_schedule_fn = linear_beta_schedule
    elif beta_schedule == "cosine":
        beta_schedule_fn = cosine_beta_schedule
    elif beta_schedule == "sigmoid":
        beta_schedule_fn = sigmoid_beta_schedule
    else:
        raise ValueError(f"unknown beta schedule {beta_schedule}")

    betas = beta_schedule_fn(timesteps)
    return betas

def extract(a, t, x_shape):
    """Extract coefficients at specified timesteps, and reshape to [batch_size, 1, 1, 1, ...] for broadcasting.

    Args:
        a: (timesteps,) tensor of coefficients.
        t: (b,) tensor of timesteps.
        x_shape: shape of the target tensor for broadcasting.
    Returns:
        out: (b, 1, 1, 1, ...) tensor of extracted coefficients.
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    out = out.reshape(b, *((1,) * (len(x_shape) - 1)))
    return out