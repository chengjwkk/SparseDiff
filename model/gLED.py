# Ref: https://github.com/FutureXiang/ddae

import torch
import torch.nn as nn
from tqdm import tqdm
from functools import partial
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast


class dsEncoder(nn.Module):
    def __init__(self, scale):
        super(dsEncoder, self).__init__()
        self.scale = scale

    def forward(self, x):
        B, C, H, W = x.shape
        H_down, W_down = int(H // self.scale), int(W // self.scale)
        x_down = F.interpolate(x, size=(H_down, W_down), mode='bilinear', align_corners=False)
        x_down_flatten = x_down.reshape(B, -1)
        return x_down_flatten # B, C*H_down*W_down


class TransformerAutoRegressive(nn.Module):
    def __init__(self, seq_len, latent_dim, n_heads, n_layers):
        super(TransformerAutoRegressive, self).__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim

        self.transformer = nn.Transformer(
            d_model=latent_dim,
            nhead=n_heads,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            batch_first=True
        )
        self.fc_out = nn.Linear(latent_dim, latent_dim)  # 用于生成下一帧的特征

        # Learnable 1D Positional Embedding
        self.positional_embedding = nn.Parameter(torch.randn(1, seq_len, latent_dim))

    def forward(self, z_input, hidden_state=None):
        """
        Args:
            z_input: Input sequence (B, T, D).
            hidden_state: Previous hidden state (B, 1, D), optional.
        Returns:
            - Predictions: Next state (B, D).
            - Updated hidden state: Last transformer output (B, 1, D).
        """
        # 拼接历史隐状态
        if hidden_state is not None:
            z_input = torch.cat([hidden_state, z_input], dim=1)  # Shape: (B, T+1, D)

        # 添加位置编码
        T = z_input.size(1)  
        pos_embed = self.positional_embedding[:, :T, :]  
        z_input = z_input + pos_embed

        transformer_output = self.transformer(z_input, z_input)

        # 获取最后一个时间步的隐状态
        last_hidden_state = transformer_output[:, -1:, :]  # Shape: (B, 1, D)
        prediction = self.fc_out(last_hidden_state)  # Shape: (B, 1, D)

        return prediction.squeeze(1), last_hidden_state  # 返回 (B, D) 和 (B, 1, D)



class transPredictor(nn.Module):
    def __init__(self, seq_len, pred_steps, scale, num_var, x_dim, y_dim, n_heads, n_layers):
        
        super(transPredictor, self).__init__()
        self.seq_len = seq_len
        self.pred_steps = pred_steps
        self.num_var = num_var
        self.x_dim = x_dim // scale
        self.y_dim = y_dim // scale
        self.latent_dim = self.x_dim * self.y_dim * num_var

        self.transformer = TransformerAutoRegressive(seq_len=seq_len, latent_dim=self.latent_dim, n_heads=n_heads, n_layers=n_layers)

    def forward(self, x):
        """
        Args:
            x:  (B, length, latent_dim)
        Returns:
             (B, lookback + per_pred_step, latent_dim)    这里应该是sample_num，返回的直接可以用，不需要在ddpm中再循环。
        """
        B, _, latent_dim = x.shape

        x_flatten = x

        hidden_state = None

        predictions = []

     
        for _ in range(self.pred_steps):
        
            prediction, hidden_state = self.transformer(x_flatten, hidden_state)

        
            predictions.append(prediction)

           
            x_flatten = prediction.unsqueeze(1)  # (B, 1, latent_dim)


        predictions = torch.cat(predictions, dim=1)

        predictions = predictions.view(B, self.pred_steps, latent_dim)
        predictions = torch.cat([x, predictions], dim=1)

        return predictions
    
    

def normalize_to_neg_one_to_one(img):
    # [0.0, 1.0] -> [-1.0, 1.0]
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    # [-1.0, 1.0] -> [0.0, 1.0]
    return (t + 1) * 0.5


def linear_beta_schedule(timesteps, beta1, beta2):
    assert 0.0 < beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"
    return torch.linspace(beta1, beta2, timesteps)


def schedules(betas, n_T, device, type='DDPM'):
    beta1, beta2 = betas
    schedule_fn = partial(linear_beta_schedule, beta1=beta1, beta2=beta2)

    if type == 'DDPM':
        beta_t = torch.cat([torch.tensor([0.0]), schedule_fn(n_T)])
    elif type == 'DDIM':
        beta_t = schedule_fn(n_T + 1)
    else:
        raise NotImplementedError()
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    ma_over_sqrtmab = (1 - alpha_t) / sqrtmab

    dic = {
        "alpha_t": alpha_t,
        "oneover_sqrta": oneover_sqrta,
        "sqrt_beta_t": sqrt_beta_t,
        "alphabar_t": alphabar_t,
        "sqrtab": sqrtab,
        "sqrtmab": sqrtmab,
        "ma_over_sqrtmab": ma_over_sqrtmab,
    }
    return {key: dic[key].to(device) for key in dic}


class GLED(nn.Module):
    def __init__(self, cUnet, cEncoder, cPredictor, betas, n_T, sche_ratios, lookback, per_pred_step, device):
        ''' DDPM proposed by "Denoising Diffusion Probabilistic Models"

            Args:
                cUnet: A network (e.g. UNet) which performs same-shape mapping.
                cEncoder: A network that encodes the residual information.
                cPredictor: A network that predicts the encoded residual information.
                device: The CUDA device that tensors run on.
            Parameters:
                betas, n_T, sche_ratios:
        '''
        super(GLED, self).__init__()
        
        self.cUnet = cUnet.to(device)
        self.cEncoder = cEncoder.to(device)
        self.cPredictor = cPredictor.to(device)
        
        self.ddpm_sche = schedules(betas, n_T, device, 'DDPM')
        self.ddim_sche = schedules(betas, n_T, device, 'DDIM')
        
        self.n_T = n_T
        self.sche_ratios = sche_ratios
        self.max_scale = len(sche_ratios) - 1
        self.per_pred_step = per_pred_step
        self.lookback = lookback
        self.device = device
        
        self.loss = nn.MSELoss()
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, use_amp=False, predict=False, e2e=False):
        ''' Training with simple noise prediction loss.

            Args:
                x: The clean image tensor ranged in `[0, 1]`, [B, 1+horizon, C, H, W].
            Returns:
                The simple MSE loss.
        '''
        B, T, C, H, W = x.shape
        x = normalize_to_neg_one_to_one(x)
        x_noised, t, noise, x_g, scale_t = self.ddpm_perturb(x, t=None)

        with autocast(enabled=use_amp):
            # Encode the conditional information
            x = x.flatten(0, 1) # B*(1+horizon), C, H, W
            zs = self.cEncoder(x) # B*(1+horizon), D
            zs = zs.reshape(B, T, -1) # B, (1+horizon), D
            
            # Predict the next horizon conditional information
            if predict:
                zs_hat = self.cPredictor(zs[:, :self.lookback].detach()) # B, 1+horizon, D
                if e2e:
                    cond_vector = zs_hat.flatten(0, 1).detach() # (1+horizon), D
                else:
                    cond_vector = zs.flatten(0, 1) # B*(1+horizon), D
                prediction_loss = self.loss(zs_hat, zs.detach())
            else:
                prediction_loss = 0.0
                cond_vector = zs.flatten(0, 1) # B*(1+horizon), D
            
            # Predict the noise
            diffusion_loss = self.loss(noise, self.cUnet(x_noised, t / self.n_T, cond_vector=cond_vector, scale=scale_t))
            return prediction_loss+diffusion_loss
              
    def ddpm_perturb(self, xs, t=None):
        ''' Add noise to a clean image (diffusion process).
            x_noised = sqrt(a_bar) * x + sqrt(1-a_bar) * noise

            Args:
                xs: The normalized image tensor.
                t: The specified timestep ranged in `[1, n_T]`. Type: int / torch.LongTensor / None.
                    Random `t ~ U[1, n_T]` is taken if t is None.
            Returns:
                The perturbed image, the corresponding timestep, and the noise.
        '''        
        # xs: B, T, C, H, W
        B, T, C, H, W = xs.shape
        
        # Random timestep
        if t is None:
            t = torch.randint(1, self.n_T + 1, (B*T, )).to(self.device)
        elif not isinstance(t, torch.Tensor):
            t = torch.tensor([t]).to(self.device).repeat(B)
        
        # Schedule
        xs = xs.flatten(0, 1) # B*T, C, H, W
        noise = torch.randn_like(xs)
        sche = self.ddpm_sche
        x_noised = (sche["sqrtab"][t, None, None, None] * xs + sche["sqrtmab"][t, None, None, None] * noise)
        scale = torch.ones(B*T, device=self.device)
        return x_noised, t, noise, None, scale
    
    # 把predictor模块挪到外面？or就在函数中定义自回归多少步
    def ddpm_sample(self, x, notqdm=False, use_amp=False):
        ''' Sampling with DDPM sampler. Actual NFE is `n_T`.

            Args:
                x: The clean image tensor ranged in `[0, 1]`.
            Returns:
                The sampled image tensor ranged in `[0, 1]`.
        '''
        # x: B, look_back, C, H, W
        assert x.shape[1] == self.lookback, "The input tensor must be lookback."
        B = x.shape[0]
        per_pred_step = self.per_pred_step
        lookback = self.lookback
        sche = self.ddpm_sche
        x = normalize_to_neg_one_to_one(x)
        
        # step1: perturb the first frame with noise
        x_i = self.ddpm_perturb(x[:,-1:], t=self.n_T)[0] # B, C, H, W
        x_i = x_i.unsqueeze(1).repeat(1, lookback+per_pred_step, 1, 1, 1) # B, per_pred_step, C, H, W
        x_i = x_i.flatten(0, 1) # B*(per_pred_step), C, H, W
        
        x = x.flatten(0, 1) # B*lookback, C, H, W
        zs = self.cEncoder(x) # B*lookback, D
        # t_eval = torch.linspace(0, self.horizon, 1+self.horizon).to(x.device)
        z0 = zs.reshape(B, lookback, -1) # B, lookback, D
        for i in range(0, per_pred_step, self.cPredictor.pred_steps):
            zs_hat = self.cPredictor(z0) if i == 0 else torch.cat([zs_hat, self.cPredictor(z0)[:,lookback:]], dim=1) # B, lookback+per_pred_step, D
            z0 = zs_hat[:, -lookback:]
        zs_hat = zs_hat[:, :lookback + per_pred_step] # B, lookback+per_pred_step, D
        cond_vector = zs_hat.flatten(0, 1) # B*(lookback+per_pred_step), D
        
        # step4: sample the frames
        n_sample, size = x_i.shape[0], x.shape[-3:]
        for i in range(self.n_T, 0, -1):
            # Timestep
            t_is = torch.tensor([i / self.n_T]).to(self.device).repeat(n_sample)
            scale = torch.ones(n_sample, device=self.device)

            # Diffusion
            z = torch.randn(n_sample, *size).to(self.device) if i > 1 else 0
            alpha = sche["alphabar_t"][i]
            eps, _ = self.pred_eps_(x_i, t_is, alpha, use_amp, cond=cond_vector, scale=scale)
            mean = sche["oneover_sqrta"][i] * (x_i - sche["ma_over_sqrtmab"][i] * eps)
            variance = sche["sqrt_beta_t"][i] # LET variance sigma_t = sqrt_beta_t
            x_i = mean + variance * z

        return unnormalize_to_zero_to_one(x_i)    # B*per_pred_step, C, H, W
    
    def pred_eps_(self, x, t, alpha, use_amp, cond, scale, clip_x=True):
        def pred_eps_from_x0(x0):
            return (x - x0 * alpha.sqrt()) / (1 - alpha).sqrt()

        def pred_x0_from_eps(eps):
            return (x - (1 - alpha).sqrt() * eps) / alpha.sqrt()

        # get prediction of x0
        with autocast(enabled=use_amp):
            # print(x.shape)
            eps = self.cUnet(x, t, cond_vector=cond, scale=scale).float()
        denoised = pred_x0_from_eps(eps)

        # pixel-space clipping (optional)
        if clip_x:
            denoised = torch.clip(denoised, -1., 1.)
            eps = pred_eps_from_x0(denoised)
        return eps, denoised