from functools import partial
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.cuda.amp import autocast as autocast
import torch.nn.functional as F
import time
import numpy as np

def normalize_to_neg_one_to_one(img):
    # [0.0, 1.0] -> [-1.0, 1.0]
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    # [-1.0, 1.0] -> [0.0, 1.0]
    return (t + 1) * 0.5


def linear_beta_schedule(timesteps, beta1, beta2):
    assert 0.0 < beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"
    return torch.linspace(beta1, beta2, timesteps)


def schedules(betas, T, device, type='DDPM'):
    beta1, beta2 = betas
    schedule_fn = partial(linear_beta_schedule, beta1=beta1, beta2=beta2)   # 提前固定一部分参数，返回固定后的函数

    if type == 'DDPM':
        beta_t = torch.cat([torch.tensor([0.0]), schedule_fn(T)])
    elif type == 'DDIM':
        beta_t = schedule_fn(T + 1)
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


def get_observation_loss(mask, square_sparse, x):
    """
    计算观测损失（Observation Loss），对每个 channel 独立归一化。

    参数：
    mask: (C, H, W)，每个通道的观测点位置
    square_sparse: (B, C, H, W)，稀疏观测值矩阵
    x: (B, C, H, W)，模型输出

    返回：
    observation_loss: (B, C, H, W)，逐点误差（已按每通道有效点数归一化）
    """
    # 误差平方
    error = (square_sparse - x) ** 2  # shape: (B, C, H, W)

    # 应用 mask（扩展成 (1, C, H, W) 以便广播）
    masked_error = error * mask.unsqueeze(0)

    # 每个 channel 的有效点数（shape: C）
    valid_points_per_channel = torch.sum(mask, dim=(1, 2))  # shape: (C,)

    # 为了广播到 (B, C, H, W)，reshape 成 (1, C, 1, 1)
    valid_points = valid_points_per_channel.view(1, -1, 1, 1) + 1e-8  # 加上小值防止除 0

    # 每个 channel 单独归一化
    observation_loss = masked_error / valid_points

    return observation_loss

def get_observation_loss_full(reconstructed, x):    # 用稀疏观测点重建后的完整图像引导
    """
    计算观测损失（Observation Loss），对每个 channel 独立归一化。

    参数：
    mask: (C, H, W)，每个通道的观测点位置
    square_sparse: (B, C, H, W)，稀疏观测值矩阵
    x: (B, C, H, W)，模型输出

    返回：
    observation_loss: (B, C, H, W)，逐点误差（已按每通道有效点数归一化）
    """
    # 误差平方
    error = (reconstructed - x) ** 2  # shape: (B, C, H, W)
    observation_loss = error / (128*128)

    return observation_loss

def laplacian(Z, dx):
    """
    计算2D拉普拉斯算子，使用周期边界条件。
    Z: 输入张量，形状为(batch_size, x_dim, y_dim)
    dx: 网格步长
    """
    # 填充边界，添加周围一圈的 0
    Z_padded = F.pad(Z, (1, 1, 1, 1), mode='constant', value=0)
    
    # 计算拉普拉斯算子：周期边界条件下的邻域加权求和
    lap = (Z_padded[:, 1:-1, :-2] + Z_padded[:, 1:-1, 2:] +
           Z_padded[:, :-2, 1:-1] + Z_padded[:, 2:, 1:-1] - 4 * Z) / dx**2
    return lap

def get_lo_pde_loss(x_cur, data_opt):
    """
    计算 LO 方程的 PDE 损失。

    参数：
    x_cur: 当前预测值 (batch_size, channels, x_dim, y_dim)，包含U和V的预测值。
    data_opt: 包含以下参数
        mu_u, mu_v: U和V的扩散系数。
        beta: 反应扩散系统的系数。
        dx: 网格步长。   

    返回：
    pde_loss: 计算得到的PDE损失 (batch_size, channels, x_dim, y_dim)。
    """
    batch_size, channels, x_dim, y_dim = x_cur.size()
    assert channels == 2, "LO系统有两个通道，U和V。"

    # 分离U和V
    U = x_cur[:, 0, :, :]
    V = x_cur[:, 1, :, :]

    # 计算拉普拉斯算子
    lap_U = laplacian(U, data_opt["dx"])
    lap_V = laplacian(V, data_opt["dx"])

    # 计算U和V的变化率（反应扩散方程右侧）
    dUdt = (1 - (U**2 + V**2)) * U + data_opt["beta"] * (U**2 + V**2) * V + data_opt["mu_u"] * lap_U
    dVdt = -data_opt["beta"] * (U**2 + V**2) * U + (1 - (U**2 + V**2)) * V + data_opt["mu_v"] * lap_V

    # 计算PDE loss：模型预测值与反应扩散方程右侧的差异
    pde_loss_U = (dUdt) ** 2
    pde_loss_V = (dVdt) ** 2

    # 创建新的损失张量，将pde_loss_U和pde_loss_V合并到一个张量中
    pde_loss = torch.stack([pde_loss_U, pde_loss_V], dim=1)

    return pde_loss  # 返回 (batch_size, channels, x_dim, y_dim)

def periodic_gradient(Z, dx):
    dZdx = (torch.roll(Z, -1, dims=2) - torch.roll(Z, 1, dims=2)) / (2 * dx)
    dZdy = (torch.roll(Z, -1, dims=1) - torch.roll(Z, 1, dims=1)) / (2 * dx)
    return dZdx, dZdy

def periodic_laplacian(Z, dx):
    return (torch.roll(Z, 1, dims=1) + torch.roll(Z, -1, dims=1) +
            torch.roll(Z, 1, dims=2) + torch.roll(Z, -1, dims=2) - 4 * Z) / dx**2

def generate_forcing(H, W, dx, shift, amplitude):
    """
    根据网格大小和参数动态生成 forcing 项。
    X, Y 用 meshgrid 生成在区间 [0, Lx], [0, Ly]
    """

    Lx, Ly = H * dx, W * dx  # dx 在两个方向一样

    x = torch.linspace(0, Lx, W)
    y = torch.linspace(0, Ly, H)
    X, Y = torch.meshgrid(x, y, indexing='xy')  # shape (W, H) × (W, H)

    F = amplitude * (torch.sin(2 * np.pi * (X + Y + shift)) +
                     torch.cos(2 * np.pi * (X + Y + shift)))  # shape (H, W)
    return F  # shape: (H, W)

def get_ns_pde_loss(x_cur, data_opt):    # 认为系统已经平衡，不引入对t的偏导
    dx = data_opt["dx"]
    nu = data_opt["nu"]
    shift = data_opt["shift"]
    amplitude = data_opt["amplitude"]

    B, channels, H, W = x_cur.shape
    assert channels == 3, "NS系统有三个通道，omega, U和V。"
    device = x_cur.device
    # 动态生成 forcing 项
    forcing = generate_forcing(H, W, dx, shift, amplitude)
    forcing = forcing.unsqueeze(0).expand(B, -1, -1).to(device)  # shape: (B, H, W)

    # 拆出 ω, u, v
    omega = x_cur[:, 0, :, :]
    u     = x_cur[:, 1, :, :]
    v     = x_cur[:, 2, :, :]

    # ===== ω 方程 =====
    omega_x, omega_y = periodic_gradient(omega, dx)
    convection_omega = -(u * omega_x + v * omega_y)
    diffusion_omega = nu * periodic_laplacian(omega, dx)
    # print(convection_omega.device, diffusion_omega.device, forcing.device)
    rhs_omega = convection_omega + diffusion_omega + forcing
    
    pde_loss_omega = (rhs_omega) ** 2   

    # ===== u 方程 =====
    u_x, u_y = periodic_gradient(u, dx)
    convection_u = -(u * u_x + v * u_y)
    diffusion_u = nu * periodic_laplacian(u, dx)
    rhs_u = convection_u + diffusion_u
    pde_loss_u = (rhs_u) ** 2

    # ===== v 方程 =====
    v_x, v_y = periodic_gradient(v, dx)
    convection_v = -(u * v_x + v * v_y)
    diffusion_v = nu * periodic_laplacian(v, dx)
    rhs_v = convection_v + diffusion_v
    pde_loss_v = (rhs_v) ** 2

    # 拼接结果
    pde_loss = torch.stack([pde_loss_omega, pde_loss_u, pde_loss_v], dim=1)
    return pde_loss  # (B, 3, H, W)


def periodic_laplacian(u, dx):
    return (
        torch.roll(u, 1, dims=-2) + torch.roll(u, -1, dims=-2) +
        torch.roll(u, 1, dims=-1) + torch.roll(u, -1, dims=-1) - 4 * u
    ) / dx**2

def periodic_biharmonic(u, dx):
    return periodic_laplacian(periodic_laplacian(u, dx), dx)

def get_sh_pde_loss(x_seq, data_opt):
    """
    x_seq: (T, 1, H, W) - sequence of predicted frames
    returns: pde_loss (T, 1, H, W)
    """
    dx = data_opt["dx"]
    dt = data_opt["dt"]
    r = data_opt["r"]
    g = data_opt["g"]

    T, _, H, W = x_seq.shape
    device = x_seq.device

    # --- Compute ∂u/∂t using conv1d ---
    # Reshape (T, 1, H, W) → (H*W, 1, T)
    u_flat = x_seq.permute(2, 3, 1, 0).reshape(-1, 1, T)  # (H*W, 1, T)

    # Central difference kernel
    deriv_kernel = torch.tensor([[[1.0, 0.0, -1.0]]], device=device, dtype=torch.float32) / (2 * dt)
    u_t_flat = F.conv1d(u_flat, deriv_kernel, padding=1)  # (H*W, 1, T)

    # Reshape back → (T, 1, H, W)
    u_t = u_t_flat.reshape(H, W, 1, T).permute(3, 2, 0, 1)

    # --- Fix boundaries: forward/backward difference ---
    u_t[0]  = (x_seq[1] - x_seq[0]) / dt
    u_t[-1] = (x_seq[-1] - x_seq[-2]) / dt

    # --- Compute RHS for each frame ---
    u = x_seq[:, 0, :, :]                         # (T, H, W)
    lap = periodic_laplacian(u, dx)               # (T, H, W)
    bih = periodic_biharmonic(u, dx)              # (T, H, W)
    rhs = r * u - 2 * lap - bih + g * u**2 - u**3

    # --- Final loss ---
    loss = (u_t[:, 0] - rhs) ** 2  # shape: (T, H, W)
    return loss.unsqueeze(1)       # shape: (T, 1, H, W)

class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device):
        ''' DDPM proposed by "Denoising Diffusion Probabilistic Models", and \
            DDIM sampler proposed by "Denoising Diffusion Implicit Models".

            Args:
                nn_model: A network (e.g. UNet) which performs same-shape mapping.
                device: The CUDA device that tensors run on.
            Parameters:
                betas, n_T
        '''
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)
        params = sum(p.numel() for p in nn_model.parameters() if p.requires_grad) / 1e6

        self.device = device
        self.ddpm_sche = schedules(betas, n_T, device, 'DDPM')
        self.ddim_sche = schedules(betas, n_T, device, 'DDIM')
        self.n_T = n_T
        self.loss = nn.MSELoss()


    def perturb(self, x, t=None):
        ''' Add noise to a clean image (diffusion process).

            Args:
                x: The normalized image tensor.
                t: The specified timestep ranged in `[1, n_T]`. Type: int / torch.LongTensor / None. \
                    Random `t ~ U[1, n_T]` is taken if t is None.
            Returns:
                The perturbed image, the corresponding timestep, and the noise.
        '''
        if t is None:
            t = torch.randint(1, self.n_T + 1, (x.shape[0], )).to(self.device)
        elif not isinstance(t, torch.Tensor):
            t = torch.tensor([t]).to(self.device).repeat(x.shape[0])

        noise = torch.randn_like(x)
        sche = self.ddpm_sche
        x_noised = (sche["sqrtab"][t, None, None, None] * x +   # broadcast
                    sche["sqrtmab"][t, None, None, None] * noise)
        return x_noised, t, noise

    def forward(self, x, use_amp=False):
        ''' Training with simple noise prediction loss.

            Args:
                x: The clean image tensor ranged in `[0, 1]`.
            Returns:
                The simple MSE loss.
        '''
        x = normalize_to_neg_one_to_one(x)
        x_noised, t, noise = self.perturb(x, t=None)

        with autocast(enabled=use_amp):
            return self.loss(noise, self.nn_model(x_noised, t / self.n_T))

    def sample(self, n_sample, size, notqdm=False, use_amp=False):
        # 二阶修正？
        # 加入引导
        ''' Sampling with DDPM sampler. Actual NFE is `n_T`.

            Args:
                n_sample: The batch size.
                size: The image shape (e.g. `(3, 32, 32)`).
            Returns:
                The sampled image tensor ranged in `[0, 1]`.
        '''
        sche = self.ddpm_sche
        x_i = torch.randn(n_sample, *size).to(self.device)   # 随机采样

        for i in tqdm(range(self.n_T, 0, -1), disable=notqdm):
            t_is = torch.tensor([i / self.n_T]).to(self.device).repeat(n_sample)  # 标准化时间步

            z = torch.randn(n_sample, *size).to(self.device) if i > 1 else 0   # 引入生成随机性，除最后一步

            alpha = sche["alphabar_t"][i]   # 系数提取
            eps, _ = self.pred_eps_(x_i, t_is, alpha, use_amp)

            mean = sche["oneover_sqrta"][i] * (x_i - sche["ma_over_sqrtmab"][i] * eps)
            variance = sche["sqrt_beta_t"][i] # LET variance sigma_t = sqrt_beta_t
            x_i = mean + variance * z

        return unnormalize_to_zero_to_one(x_i)
    
    def ddpm_guided_sample(self, n_sample, size, zeta_obs, zeta_pde, ratio, proportion, mask, square_sparse, data_opt, notqdm=False):
        # 二阶修正？
        # 加入引导
        ''' Sampling with DDPM sampler. Actual NFE is `n_T`.

            Args:
                n_sample: The batch size.
                size: The image shape (e.g. `(3, 32, 32)`).
            Returns:
                The sampled image tensor ranged in `[0, 1]`.
        '''
      
        sche = self.ddpm_sche
        x_next = torch.randn(n_sample, *size).to(self.device)   # 随机采样
       
        mask, square_sparse = mask.to(self.device), normalize_to_neg_one_to_one(square_sparse).to(self.device)

        for i in range(self.n_T, 0, -1):   # 需要记录x_cur 和 x_next
            x_cur = x_next.detach().clone()
            x_cur.requires_grad_(True)
            # print(x_cur.grad_fn)
            t_is = torch.tensor([i / self.n_T]).to(self.device).repeat(n_sample)  # 标准化时间步

            z = torch.randn(n_sample, *size).to(self.device) if i > 1 else 0   # 引入生成随机性，除最后一步

            alpha = sche["alphabar_t"][i]   # 系数提取
            eps, _ = self.pred_eps_(x_cur, t_is, alpha)

            mean = sche["oneover_sqrta"][i] * (x_cur - sche["ma_over_sqrtmab"][i] * eps)
            variance = sche["sqrt_beta_t"][i] # LET variance sigma_t = sqrt_beta_t
            x_next = mean + variance * z
            
            observation_loss = get_observation_loss(mask, square_sparse, x_cur)
            pde_loss = get_lo_pde_loss(x_cur, data_opt)/(128*128)    # 系数应该都是给定的，dx也。（数据集一旦确定）
                
                # # 参数定义
                # mu_u = 0.1
                # mu_v = 0.1
                # beta = 1.0
                # time_step = 0.1
                # total_time = 100.0
                # nx, ny = 64, 64  # 网格大小
                # Lx, Ly = 20, 20  # 空间范围 [-10, 10]

                # # 定义网格步长
                # dx = Lx / (nx - 1)
                # dy = Ly / (ny - 1)
            # print("x_cur.requires_grad:", x_cur.requires_grad)
            # print("observation_loss.requires_grad:", observation_loss.requires_grad)
            # print("observation_loss.grad_fn:", observation_loss.grad_fn)
            # print("observation_loss shape:", observation_loss.shape)
        
            grad_x_cur_obs = torch.autograd.grad(outputs=observation_loss.sum(), inputs=x_cur, retain_graph=True)[0]   # (B, channels, 128, 128)
            grad_x_cur_pde = torch.autograd.grad(outputs=pde_loss.sum(), inputs=x_cur, retain_graph=True)[0]   # (B, channels, 128, 128)
            if i >= proportion * self.n_T:
                x_next = x_next - zeta_obs * grad_x_cur_obs
                # print("obs min:",grad_x_cur_obs.min().item(), "max:",grad_x_cur_obs.max().item(), "mean:", grad_x_cur_obs.mean().item())
                
            else:
                x_next = x_next - ratio * (zeta_obs * grad_x_cur_obs) - zeta_pde * grad_x_cur_pde
                # print("pde min:",grad_x_cur_pde.min().item(), "max:",grad_x_cur_pde.max().item(), "mean:", grad_x_cur_pde.mean().item())

        return unnormalize_to_zero_to_one(x_next)
    
    def ddpm_guided_sample_full(self, n_sample, size, zeta_obs, zeta_pde, ratio, reconstructed, data_opt, notqdm=False):
       
        ''' Sampling with DDPM sampler. Actual NFE is `n_T`.

            Args:
                n_sample: The batch size.
                size: The image shape (e.g. `(3, 32, 32)`).
            Returns:
                The sampled image tensor ranged in `[0, 1]`.
        '''
      
        sche = self.ddpm_sche
        x_next = torch.randn(n_sample, *size).to(self.device)   # 随机采样
       
        reconstructed = normalize_to_neg_one_to_one(reconstructed).to(self.device)

        for i in tqdm(range(self.n_T, 0, -1), disable=notqdm):   # 需要记录x_cur 和 x_next
            x_cur = x_next.detach().clone()
            x_cur.requires_grad_(True)
            # print(x_cur.grad_fn)
            t_is = torch.tensor([i / self.n_T]).to(self.device).repeat(n_sample)  # 标准化时间步

            z = torch.randn(n_sample, *size).to(self.device) if i > 1 else 0   # 引入生成随机性，除最后一步

            alpha = sche["alphabar_t"][i]   
            
            eps, _ = self.pred_eps_(x_cur, t_is, alpha)

            mean = sche["oneover_sqrta"][i] * (x_cur - sche["ma_over_sqrtmab"][i] * eps)
            variance = sche["sqrt_beta_t"][i] # LET variance sigma_t = sqrt_beta_t
            x_next = mean + variance * z
            
            observation_loss = get_observation_loss_full(reconstructed, x_cur)
            
               
        
            grad_x_cur_obs = torch.autograd.grad(outputs=observation_loss.sum(), inputs=x_cur, retain_graph=True)[0]   # (B, channels, 128, 128)
            
            if i >= 0.2 * self.n_T:
                x_next = x_next - zeta_obs * grad_x_cur_obs
                # print("obs min:",grad_x_cur_obs.min().item(), "max:",grad_x_cur_obs.max().item(), "mean:", grad_x_cur_obs.mean().item())
                
            else:
                pde_loss = get_lo_pde_loss(x_cur, data_opt)/(128*128)    # 系数应该都是给定的，dx也。（数据集一旦确定）
                grad_x_cur_pde = torch.autograd.grad(outputs=pde_loss.sum(), inputs=x_cur, retain_graph=True)[0]   # (B, channels, 128, 128)
                x_next = x_next - 0.1 * (zeta_obs * grad_x_cur_obs) - zeta_pde * grad_x_cur_pde
                # print("pde min:",grad_x_cur_pde.min().item(), "max:",grad_x_cur_pde.max().item(), "mean:", grad_x_cur_pde.mean().item())

        return unnormalize_to_zero_to_one(x_next)
    
    def sample_from_reconstructed_lo(self, n_sample, size, zeta_pde, ratio, reconstructed, T, data_opt, notqdm=False):
       
        ''' Sampling with DDPM sampler. Actual NFE is `n_T`.

            Args:
                n_sample: The batch size.
                size: The image shape (e.g. `(3, 32, 32)`).
            Returns:
                The sampled image tensor ranged in `[0, 1]`.
        '''
      
        sche = self.ddpm_sche
        # 直接从重建开始去噪
        
       
        reconstructed = normalize_to_neg_one_to_one(reconstructed).to(self.device)
        x_next = reconstructed.to(self.device)   # 随机采样

        for i in tqdm(range(T, 0, -1), disable=notqdm):   # 需要记录x_cur 和 x_next
            x_cur = x_next.detach().clone()
            x_cur.requires_grad_(True)
            # print(x_cur.grad_fn)
            # t_is = torch.tensor([i / self.n_T]).to(self.device).repeat(n_sample)  # 标准化时间步

            # z = torch.randn(n_sample, *size).to(self.device) if i > 1 else 0   # 引入生成随机性，除最后一步

            # alpha = sche["alphabar_t"][i]   # 系数提取
            # eps, _ = self.pred_eps_(x_cur, t_is, alpha)

            # mean = sche["oneover_sqrta"][i] * (x_cur - sche["ma_over_sqrtmab"][i] * eps)
            # variance = sche["sqrt_beta_t"][i] # LET variance sigma_t = sqrt_beta_t
            # x_next = mean + variance * z
            
            x_next = x_cur
            
            # observation_loss = get_observation_loss_full(reconstructed, x_cur)
            
               
        
            # grad_x_cur_obs = torch.autograd.grad(outputs=observation_loss.sum(), inputs=x_cur, retain_graph=True)[0]   # (B, channels, 128, 128)
            
            # if i >= 0.2 * self.n_T:
            #     x_next = x_next - zeta_obs * grad_x_cur_obs
            #     # print("obs min:",grad_x_cur_obs.min().item(), "max:",grad_x_cur_obs.max().item(), "mean:", grad_x_cur_obs.mean().item())
                
            # else:
            
            pde_loss = get_lo_pde_loss(x_cur, data_opt)/(128*128)    # 系数应该都是给定的，dx也。（数据集一旦确定）
            grad_x_cur_pde = torch.autograd.grad(outputs=pde_loss.sum(), inputs=x_cur, retain_graph=True)[0]   # (B, channels, 128, 128)
            x_next = x_next - zeta_pde * grad_x_cur_pde
            # print("pde min:",grad_x_cur_pde.min().item(), "max:",grad_x_cur_pde.max().item(), "mean:", grad_x_cur_pde.mean().item())

        return unnormalize_to_zero_to_one(x_next)

    def ddpm_guided_sample_full_time(self, n_sample, size, zeta_obs, zeta_pde, ratio, reconstructed, data_opt, notqdm=False):
        sche = self.ddpm_sche
        x_next = torch.randn(n_sample, *size).to(self.device)
        reconstructed = normalize_to_neg_one_to_one(reconstructed).to(self.device)

        for i in tqdm(range(self.n_T, 0, -1), disable=notqdm):
            t_start = time.time()

            x_cur = x_next.detach().clone()
            x_cur.requires_grad_(True)

            t_is = torch.tensor([i / self.n_T]).to(self.device).repeat(n_sample)
            z = torch.randn(n_sample, *size).to(self.device) if i > 1 else 0

            alpha = sche["alphabar_t"][i]

            # --- pred_eps
            t1 = time.time()
            eps, _ = self.pred_eps_(x_cur, t_is, alpha)
            t2 = time.time()

            mean = sche["oneover_sqrta"][i] * (x_cur - sche["ma_over_sqrtmab"][i] * eps)
            variance = sche["sqrt_beta_t"][i]
            x_next = mean + variance * z

            # --- observation loss
            t3 = time.time()
            # observation_loss = get_observation_loss_full(reconstructed, x_cur)
            t4 = time.time()

            # --- grad_obs
            # grad_x_cur_obs = torch.autograd.grad(outputs=observation_loss.sum(), inputs=x_cur, retain_graph=True)[0]
            t5 = time.time()

            if i >= 0.2 * self.n_T:
                # --- update step without pde
                # x_next = x_next - zeta_obs * grad_x_cur_obs
                t6 = time.time()
                # print(f"[Step {i:4d}] pred_eps: {t2 - t1:.3f}s | obs_loss: {t4 - t3:.3f}s | grad_obs: {t5 - t4:.3f}s | update: {t6 - t5:.3f}s | total: {t6 - t_start:.3f}s")
            else:
                # --- pde loss
                t6 = time.time()
                # pde_loss = get_lo_pde_loss(x_cur, data_opt)/(128*128)
                t7 = time.time()

                # --- grad_pde
                # grad_x_cur_pde = torch.autograd.grad(outputs=pde_loss.sum(), inputs=x_cur, retain_graph=True)[0]
                t8 = time.time()

                # x_next = x_next - 0.1 * (zeta_obs * grad_x_cur_obs) - zeta_pde * grad_x_cur_pde
                t9 = time.time()

                # print(f"[Step {i:4d}] pred_eps: {t2 - t1:.3f}s | obs_loss: {t4 - t3:.3f}s | grad_obs: {t5 - t4:.3f}s | pde_loss: {t7 - t6:.3f}s | grad_pde: {t8 - t7:.3f}s | update: {t9 - t8:.3f}s | total: {t9 - t_start:.3f}s")

        return unnormalize_to_zero_to_one(x_next)
        
    def ddpm_guided_sample_full_cond(self, n_sample, size, zeta_obs, zeta_pde, ratio, reconstructed, data_opt, notqdm=False):
        # 二阶修正？
        # 加入引导
        ''' Sampling with DDPM sampler. Actual NFE is `n_T`.

            Args:
                n_sample: The batch size.
                size: The image shape (e.g. `(3, 32, 32)`).
            Returns:
                The sampled image tensor ranged in `[0, 1]`.
        '''
      
        sche = self.ddpm_sche
        x_next = torch.randn(n_sample, *size).to(self.device)   # 随机采样
       
        reconstructed = normalize_to_neg_one_to_one(reconstructed).to(self.device)

        for i in range(self.n_T, 0, -1):   # 需要记录x_cur 和 x_next
            x_cur = x_next.detach().clone()
            x_cur.requires_grad_(True)
            # print(x_cur.grad_fn)
            t_is = torch.tensor([i / self.n_T]).to(self.device).repeat(n_sample)  # 标准化时间步

            z = torch.randn(n_sample, *size).to(self.device) if i > 1 else 0   # 引入生成随机性，除最后一步

            alpha = sche["alphabar_t"][i]   # 系数提取
            eps, _ = self.pred_eps_cond(x_cur, t_is, alpha, reconstructed)

            mean = sche["oneover_sqrta"][i] * (x_cur - sche["ma_over_sqrtmab"][i] * eps)
            variance = sche["sqrt_beta_t"][i] # LET variance sigma_t = sqrt_beta_t
            x_next = mean + variance * z
            
            observation_loss = get_observation_loss_full(reconstructed, x_cur)
            
              
        
            grad_x_cur_obs = torch.autograd.grad(outputs=observation_loss.sum(), inputs=x_cur, retain_graph=True)[0]   # (B, channels, 128, 128)
            
            if i >= 0.2 * self.n_T:
                x_next = x_next - zeta_obs * grad_x_cur_obs
                # print("obs min:",grad_x_cur_obs.min().item(), "max:",grad_x_cur_obs.max().item(), "mean:", grad_x_cur_obs.mean().item())
                
            else:
                pde_loss = get_lo_pde_loss(x_cur, data_opt)/(128*128)    # 系数应该都是给定的，dx也。（数据集一旦确定）
                grad_x_cur_pde = torch.autograd.grad(outputs=pde_loss.sum(), inputs=x_cur, retain_graph=True)[0]   # (B, channels, 128, 128)
                x_next = x_next - 0.1 * (zeta_obs * grad_x_cur_obs) - zeta_pde * grad_x_cur_pde
                # print("pde min:",grad_x_cur_pde.min().item(), "max:",grad_x_cur_pde.max().item(), "mean:", grad_x_cur_pde.mean().item())

        return unnormalize_to_zero_to_one(x_next)

    def ddim_guided_sample_full_lo(self, n_sample, size, steps=100, eta=0.0, 
                        zeta_obs=0.0, zeta_pde=0.0, ratio = 0.1, 
                        reconstructed=None, data_opt=None, notqdm=False):
        '''
        Guided sampling with DDIM.

        Args:
            n_sample: batch size
            size: (C, H, W)
            steps: total number of DDIM steps (<< self.n_T)
            eta: noise scale (eta=0.0 is deterministic DDIM)
            zeta_obs: weight of observation gradient
            zeta_pde: weight of PDE gradient
            reconstructed: observation image for loss guidance (must be normalized to [0, 1])
            data_opt: additional data for PDE loss
        '''
        sche = self.ddim_sche
        x_i = torch.randn(n_sample, *size).to(self.device)

        if reconstructed is not None:
            reconstructed = normalize_to_neg_one_to_one(reconstructed).to(self.device)

        # Subsample time steps
        times = torch.arange(0, self.n_T, self.n_T // steps) + 1
        times = list(reversed(times.int().tolist())) + [0]
        time_pairs = list(zip(times[:-1], times[1:]))

        for time, time_next in tqdm(time_pairs, disable=notqdm):
            x_i = x_i.detach().clone()
            x_i.requires_grad_(True)

            t_is = torch.tensor([time / self.n_T], device=self.device).repeat(n_sample)
            z = torch.randn(n_sample, *size).to(self.device) if (eta > 0 and time_next > 0) else 0

            # Predict x0 and eps
            
            alpha = sche["alphabar_t"][time]
            eps, x0_t = self.pred_eps_(x_i, t_is, alpha)

            alpha_next = sche["alphabar_t"][time_next]
            c1 = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c2 = (1 - alpha_next - c1 ** 2).sqrt()

            x_next = alpha_next.sqrt() * x0_t + c2 * eps + c1 * z

            # === Gradient guidance ===
            if reconstructed is not None:
                obs_loss = get_observation_loss_full(reconstructed, x_i)
                grad_obs = torch.autograd.grad(obs_loss.sum(), x_i, retain_graph=True)[0]

                if time > 0.2 * self.n_T:
                    x_next = x_next - zeta_obs * grad_obs
                else:
                    pde_loss = get_lo_pde_loss(x_i, data_opt) / (128 * 128)
                    grad_pde = torch.autograd.grad(pde_loss.sum(), x_i, retain_graph=True)[0]
                    x_next = x_next - 0.1 * (zeta_obs * grad_obs) - zeta_pde * grad_pde

            x_i = x_next

        return unnormalize_to_zero_to_one(x_i)
    
    def ddim_guided_sample_full_sh(self, n_sample, size, steps=100, eta=0.0, 
                        zeta_obs=0.0, zeta_pde=0.0, ratio = 0.1, 
                        reconstructed=None, data_opt=None, notqdm=False):
        '''
        Guided sampling with DDIM.

        Args:
            n_sample: batch size
            size: (C, H, W)
            steps: total number of DDIM steps (<< self.n_T)
            eta: noise scale (eta=0.0 is deterministic DDIM)
            zeta_obs: weight of observation gradient
            zeta_pde: weight of PDE gradient
            reconstructed: observation image for loss guidance (must be normalized to [0, 1])
            data_opt: additional data for PDE loss
        '''
        sche = self.ddim_sche
        x_i = torch.randn(n_sample, *size).to(self.device)

        if reconstructed is not None:
            reconstructed = normalize_to_neg_one_to_one(reconstructed).to(self.device)

        # Subsample time steps
        times = torch.arange(0, self.n_T, self.n_T // steps) + 1
        times = list(reversed(times.int().tolist())) + [0]
        time_pairs = list(zip(times[:-1], times[1:]))

        for time, time_next in tqdm(time_pairs, disable=notqdm):
            x_i = x_i.detach().clone()
            x_i.requires_grad_(True)

            t_is = torch.tensor([time / self.n_T], device=self.device).repeat(n_sample)
            z = torch.randn(n_sample, *size).to(self.device) if (eta > 0 and time_next > 0) else 0

            # Predict x0 and eps
            alpha = sche["alphabar_t"][time]
            eps, x0_t = self.pred_eps_(x_i, t_is, alpha)

            alpha_next = sche["alphabar_t"][time_next]
            c1 = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c2 = (1 - alpha_next - c1 ** 2).sqrt()

            x_next = alpha_next.sqrt() * x0_t + c2 * eps + c1 * z

            # === Gradient guidance ===
            if reconstructed is not None:
                obs_loss = get_observation_loss_full(reconstructed, x_i)
                grad_obs = torch.autograd.grad(obs_loss.sum(), x_i, retain_graph=True)[0]

                if time > 0.2 * self.n_T:
                    x_next = x_next - zeta_obs * grad_obs
                else:
                    pde_loss = get_sh_pde_loss(x_i, data_opt) / (128 * 128)
                    grad_pde = torch.autograd.grad(pde_loss.sum(), x_i, retain_graph=True)[0]
                    x_next = x_next - 0.1 * (zeta_obs * grad_obs) - zeta_pde * grad_pde

            x_i = x_next

        return unnormalize_to_zero_to_one(x_i)
    
    def ddim_guided_sample_full_cy(self, n_sample, size, steps=100, eta=0.0, 
                        zeta_obs=0.0, zeta_pde=0.0, ratio = 0.1, 
                        reconstructed=None, data_opt=None, notqdm=False):
        '''
        Guided sampling with DDIM.

        Args:
            n_sample: batch size
            size: (C, H, W)
            steps: total number of DDIM steps (<< self.n_T)
            eta: noise scale (eta=0.0 is deterministic DDIM)
            zeta_obs: weight of observation gradient
            zeta_pde: weight of PDE gradient
            reconstructed: observation image for loss guidance (must be normalized to [0, 1])
            data_opt: additional data for PDE loss
        '''
        sche = self.ddim_sche
        x_i = torch.randn(n_sample, *size).to(self.device)

        if reconstructed is not None:
            reconstructed = normalize_to_neg_one_to_one(reconstructed).to(self.device)

        # Subsample time steps
        times = torch.arange(0, self.n_T, self.n_T // steps) + 1
        times = list(reversed(times.int().tolist())) + [0]
        time_pairs = list(zip(times[:-1], times[1:]))

        for time, time_next in tqdm(time_pairs, disable=notqdm):
            x_i = x_i.detach().clone()
            x_i.requires_grad_(True)

            t_is = torch.tensor([time / self.n_T], device=self.device).repeat(n_sample)
            z = torch.randn(n_sample, *size).to(self.device) if (eta > 0 and time_next > 0) else 0

            # Predict x0 and eps
            alpha = sche["alphabar_t"][time]
            eps, x0_t = self.pred_eps_(x_i, t_is, alpha)

            alpha_next = sche["alphabar_t"][time_next]
            c1 = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c2 = (1 - alpha_next - c1 ** 2).sqrt()

            x_next = alpha_next.sqrt() * x0_t + c2 * eps + c1 * z

            # === Gradient guidance ===
            if reconstructed is not None:
                obs_loss = get_observation_loss_full(reconstructed, x_i)
                grad_obs = torch.autograd.grad(obs_loss.sum(), x_i, retain_graph=True)[0]

                if time > 0.2 * self.n_T:
                    x_next = x_next - zeta_obs * grad_obs
                else:
                    # pde_loss = get_cy_pde_loss(x_i, data_opt) / (128 * 128)
                    # grad_pde = torch.autograd.grad(pde_loss.sum(), x_i, retain_graph=True)[0]
                    x_next = x_next - 0.1 * (zeta_obs * grad_obs) 

            x_i = x_next

        return unnormalize_to_zero_to_one(x_i)
    
    def ddim_guided_sample_full_sevir_tem(self, n_sample, size, steps=100, eta=0.0, 
                        zeta_obs=0.0, zeta_pde=0.0, ratio = 0.1, 
                        reconstructed=None, data_opt=None, notqdm=False):
        '''
        Guided sampling with DDIM.

        Args:
            n_sample: batch size
            size: (C, H, W)
            steps: total number of DDIM steps (<< self.n_T)
            eta: noise scale (eta=0.0 is deterministic DDIM)
            zeta_obs: weight of observation gradient
            zeta_pde: weight of PDE gradient
            reconstructed: observation image for loss guidance (must be normalized to [0, 1])
            data_opt: additional data for PDE loss
        '''
        sche = self.ddim_sche
        x_i = torch.randn(n_sample, *size).to(self.device)

        if reconstructed is not None:
            reconstructed = normalize_to_neg_one_to_one(reconstructed).to(self.device)

        # Subsample time steps
        times = torch.arange(0, self.n_T, self.n_T // steps) + 1
        times = list(reversed(times.int().tolist())) + [0]
        time_pairs = list(zip(times[:-1], times[1:]))

        for time, time_next in tqdm(time_pairs, disable=notqdm):
            x_i = x_i.detach().clone()
            x_i.requires_grad_(True)

            t_is = torch.tensor([time / self.n_T], device=self.device).repeat(n_sample)
            z = torch.randn(n_sample, *size).to(self.device) if (eta > 0 and time_next > 0) else 0

            # Predict x0 and eps
            alpha = sche["alphabar_t"][time]
            eps, x0_t = self.pred_eps_(x_i, t_is, alpha)

            alpha_next = sche["alphabar_t"][time_next]
            c1 = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c2 = (1 - alpha_next - c1 ** 2).sqrt()

            x_next = alpha_next.sqrt() * x0_t + c2 * eps + c1 * z

            # === Gradient guidance ===
            if reconstructed is not None:
                obs_loss = get_observation_loss_full(reconstructed, x_i)
                grad_obs = torch.autograd.grad(obs_loss.sum(), x_i, retain_graph=True)[0]

                if time > 0.2 * self.n_T:
                    x_next = x_next - zeta_obs * grad_obs
                else:
                    # pde_loss = get_cy_pde_loss(x_i, data_opt) / (128 * 128)
                    # grad_pde = torch.autograd.grad(pde_loss.sum(), x_i, retain_graph=True)[0]
                    x_next = x_next - 0.1 * (zeta_obs * grad_obs) 

            x_i = x_next

        return unnormalize_to_zero_to_one(x_i)
    
    def ddim_sample_from_reconstructed_lo(self, n_sample, size, steps=100, eta=0.0, 
                        zeta_obs=0.0, zeta_pde=0.0, ratio = 0.1, 
                        reconstructed=None, data_opt=None, notqdm=False):
        '''
        Guided sampling with DDIM.

        Args:
            n_sample: batch size
            size: (C, H, W)
            steps: total number of DDIM steps (<< self.n_T)
            eta: noise scale (eta=0.0 is deterministic DDIM)
            zeta_obs: weight of observation gradient
            zeta_pde: weight of PDE gradient
            reconstructed: observation image for loss guidance (must be normalized to [0, 1])
            data_opt: additional data for PDE loss
        '''
        sche = self.ddim_sche
        

        if reconstructed is not None:
            reconstructed = normalize_to_neg_one_to_one(reconstructed).to(self.device)
        x_i = reconstructed

        # Subsample time steps
        # 认为现有的图像已经是去噪到50步的结果？
        times = torch.arange(0, self.n_T/20, self.n_T/20 // steps) + 1
        times = list(reversed(times.int().tolist())) + [0]
        time_pairs = list(zip(times[:-1], times[1:]))
        # print(time_pairs)

        for time, time_next in tqdm(time_pairs, disable=notqdm):
            x_i = x_i.detach().clone()
            x_i.requires_grad_(True)

            t_is = torch.tensor([time / self.n_T], device=self.device).repeat(n_sample)
            z = torch.randn(n_sample, *size).to(self.device) if (eta > 0 and time_next > 0) else 0

            # Predict x0 and eps
            alpha = sche["alphabar_t"][time]
            eps, x0_t = self.pred_eps_(x_i, t_is, alpha)

            alpha_next = sche["alphabar_t"][time_next]
            c1 = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c2 = (1 - alpha_next - c1 ** 2).sqrt()

            x_next = alpha_next.sqrt() * x0_t + c2 * eps + c1 * z

            # === Gradient guidance ===
            if reconstructed is not None:
                pde_loss = get_lo_pde_loss(x_i, data_opt) / (128 * 128)
                grad_pde = torch.autograd.grad(pde_loss.sum(), x_i, retain_graph=True)[0]
                x_next = x_next - zeta_pde * grad_pde

            x_i = x_next

        return unnormalize_to_zero_to_one(x_i)
    
    def ddim_sample_from_reconstructed_ns(self, n_sample, size, steps=100, eta=0.0, 
                        zeta_obs=0.0, zeta_pde=0.0, ratio = 0.1, 
                        reconstructed=None, data_opt=None, notqdm=False):
        '''
        Guided sampling with DDIM.

        Args:
            n_sample: batch size
            size: (C, H, W)
            steps: total number of DDIM steps (<< self.n_T)
            eta: noise scale (eta=0.0 is deterministic DDIM)
            zeta_obs: weight of observation gradient
            zeta_pde: weight of PDE gradient
            reconstructed: observation image for loss guidance (must be normalized to [0, 1])
            data_opt: additional data for PDE loss
        '''
        sche = self.ddim_sche
        

        if reconstructed is not None:
            reconstructed = normalize_to_neg_one_to_one(reconstructed).to(self.device)
        x_i = reconstructed

        # Subsample time steps
        # 认为现有的图像已经是去噪到50步的结果？
        times = torch.arange(0, self.n_T/20, self.n_T/20 // steps) + 1
        times = list(reversed(times.int().tolist())) + [0]
        time_pairs = list(zip(times[:-1], times[1:]))
        # print(time_pairs)

        for time, time_next in tqdm(time_pairs, disable=notqdm):
            x_i = x_i.detach().clone()
            x_i.requires_grad_(True)

            t_is = torch.tensor([time / self.n_T], device=self.device).repeat(n_sample)
            z = torch.randn(n_sample, *size).to(self.device) if (eta > 0 and time_next > 0) else 0

            # Predict x0 and eps
            alpha = sche["alphabar_t"][time]
            eps, x0_t = self.pred_eps_(x_i, t_is, alpha)

            alpha_next = sche["alphabar_t"][time_next]
            c1 = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c2 = (1 - alpha_next - c1 ** 2).sqrt()

            x_next = alpha_next.sqrt() * x0_t + c2 * eps + c1 * z

            # === Gradient guidance ===
            if reconstructed is not None:
                pde_loss = get_ns_pde_loss(x_i, data_opt) / (128 * 128)
                grad_pde = torch.autograd.grad(pde_loss.sum(), x_i, retain_graph=True)[0]
                x_next = x_next - zeta_pde * grad_pde

            x_i = x_next

        return unnormalize_to_zero_to_one(x_i)
    
    def ddim_sample_from_reconstructed_sh(self, n_sample, size, steps=100, eta=0.0, 
                        zeta_obs=0.0, zeta_pde=0.0, ratio = 0.1, 
                        reconstructed=None, data_opt=None, notqdm=False):
        '''
        Guided sampling with DDIM.

        Args:
            n_sample: batch size
            size: (C, H, W)
            steps: total number of DDIM steps (<< self.n_T)
            eta: noise scale (eta=0.0 is deterministic DDIM)
            zeta_obs: weight of observation gradient
            zeta_pde: weight of PDE gradient
            reconstructed: observation image for loss guidance (must be normalized to [0, 1])
            data_opt: additional data for PDE loss
        '''
        sche = self.ddim_sche
        

        if reconstructed is not None:
            reconstructed = normalize_to_neg_one_to_one(reconstructed).to(self.device)
        x_i = reconstructed

        # Subsample time steps
        # 认为现有的图像已经是去噪到50步的结果？
        times = torch.arange(0, self.n_T/20, self.n_T/20 // steps) + 1
        times = list(reversed(times.int().tolist())) + [0]
        time_pairs = list(zip(times[:-1], times[1:]))
        # print(time_pairs)

        for time, time_next in tqdm(time_pairs, disable=notqdm):
            x_i = x_i.detach().clone()
            x_i.requires_grad_(True)

            t_is = torch.tensor([time / self.n_T], device=self.device).repeat(n_sample)
            z = torch.randn(n_sample, *size).to(self.device) if (eta > 0 and time_next > 0) else 0

            # Predict x0 and eps
            alpha = sche["alphabar_t"][time]
            eps, x0_t = self.pred_eps_(x_i, t_is, alpha)

            alpha_next = sche["alphabar_t"][time_next]
            c1 = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c2 = (1 - alpha_next - c1 ** 2).sqrt()

            x_next = alpha_next.sqrt() * x0_t + c2 * eps + c1 * z

            # === Gradient guidance ===
            if reconstructed is not None:
                pde_loss = get_sh_pde_loss(x_i, data_opt) / (128 * 128)
                grad_pde = torch.autograd.grad(pde_loss.sum(), x_i, retain_graph=True)[0]
                x_next = x_next - zeta_pde * grad_pde

            x_i = x_next

        return unnormalize_to_zero_to_one(x_i)
    
    def ddim_sample_from_reconstructed_sevir_tem(self, n_sample, size, steps=100, eta=0.0, 
                        zeta_obs=0.0, zeta_pde=0.0, ratio = 0.1, 
                        reconstructed=None, data_opt=None, notqdm=False):
        '''
        Guided sampling with DDIM.

        Args:
            n_sample: batch size
            size: (C, H, W)
            steps: total number of DDIM steps (<< self.n_T)
            eta: noise scale (eta=0.0 is deterministic DDIM)
            zeta_obs: weight of observation gradient
            zeta_pde: weight of PDE gradient
            reconstructed: observation image for loss guidance (must be normalized to [0, 1])
            data_opt: additional data for PDE loss
        '''
        sche = self.ddim_sche
        

        if reconstructed is not None:
            reconstructed = normalize_to_neg_one_to_one(reconstructed).to(self.device)
        x_i = reconstructed

        # Subsample time steps
        # 认为现有的图像已经是去噪到50步的结果？
        times = torch.arange(0, self.n_T/20, self.n_T/20 // steps) + 1
        times = list(reversed(times.int().tolist())) + [0]
        time_pairs = list(zip(times[:-1], times[1:]))
        # print(time_pairs)

        for time, time_next in tqdm(time_pairs, disable=notqdm):
            x_i = x_i.detach().clone()
            x_i.requires_grad_(True)

            t_is = torch.tensor([time / self.n_T], device=self.device).repeat(n_sample)
            z = torch.randn(n_sample, *size).to(self.device) if (eta > 0 and time_next > 0) else 0

            # Predict x0 and eps
            alpha = sche["alphabar_t"][time]
            eps, x0_t = self.pred_eps_(x_i, t_is, alpha)

            alpha_next = sche["alphabar_t"][time_next]
            c1 = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c2 = (1 - alpha_next - c1 ** 2).sqrt()

            x_next = alpha_next.sqrt() * x0_t + c2 * eps + c1 * z

            # # === Gradient guidance ===
            # if reconstructed is not None:
            #     pde_loss = get_sh_pde_loss(x_i, data_opt) / (128 * 128)
            #     grad_pde = torch.autograd.grad(pde_loss.sum(), x_i, retain_graph=True)[0]
            #     x_next = x_next - zeta_pde * grad_pde

            x_i = x_next

        return unnormalize_to_zero_to_one(x_i)
    
    def ddim_guided_sample_full_ns(self, n_sample, size, steps=100, eta=0.0, 
                        zeta_obs=0.0, zeta_pde=0.0, ratio = 0.1, 
                        reconstructed=None, data_opt=None, notqdm=False):
        '''
        Guided sampling with DDIM.

        Args:
            n_sample: batch size
            size: (C, H, W)
            steps: total number of DDIM steps (<< self.n_T)
            eta: noise scale (eta=0.0 is deterministic DDIM)
            zeta_obs: weight of observation gradient
            zeta_pde: weight of PDE gradient
            reconstructed: observation image for loss guidance (must be normalized to [0, 1])
            data_opt: additional data for PDE loss
        '''
        sche = self.ddim_sche
        x_i = torch.randn(n_sample, *size).to(self.device)

        if reconstructed is not None:
            reconstructed = normalize_to_neg_one_to_one(reconstructed).to(self.device)

        # Subsample time steps
        times = torch.arange(0, self.n_T, self.n_T // steps) + 1
        times = list(reversed(times.int().tolist())) + [0]
        time_pairs = list(zip(times[:-1], times[1:]))

        for time, time_next in tqdm(time_pairs, disable=notqdm):
            x_i = x_i.detach().clone()
            x_i.requires_grad_(True)

            t_is = torch.tensor([time / self.n_T], device=self.device).repeat(n_sample)
            z = torch.randn(n_sample, *size).to(self.device) if (eta > 0 and time_next > 0) else 0

            # Predict x0 and eps
            alpha = sche["alphabar_t"][time]
            eps, x0_t = self.pred_eps_(x_i, t_is, alpha)

            alpha_next = sche["alphabar_t"][time_next]
            c1 = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c2 = (1 - alpha_next - c1 ** 2).sqrt()

            x_next = alpha_next.sqrt() * x0_t + c2 * eps + c1 * z

            # === Gradient guidance ===
            if reconstructed is not None:
                obs_loss = get_observation_loss_full(reconstructed, x_i)
                grad_obs = torch.autograd.grad(obs_loss.sum(), x_i, retain_graph=True)[0]

                if time > 0.2 * self.n_T:
                    x_next = x_next - zeta_obs * grad_obs
                else:
                    pde_loss = get_ns_pde_loss(x_i, data_opt) / (128 * 128)
                    grad_pde = torch.autograd.grad(pde_loss.sum(), x_i, retain_graph=True)[0]
                    x_next = x_next - 0.1 * (zeta_obs * grad_obs) - zeta_pde * grad_pde

            x_i = x_next

        return unnormalize_to_zero_to_one(x_i)
    
    def ddim_sample(self, n_sample, size, steps=100, eta=0.0, notqdm=False, use_amp=False):
        ''' Sampling with DDIM sampler. Actual NFE is `steps`.

            Args:
                n_sample: The batch size.
                size: The image shape (e.g. `(3, 32, 32)`).
                steps: The number of total timesteps.
                eta: controls stochasticity. Set `eta=0` for deterministic sampling.
            Returns:
                The sampled image tensor ranged in `[0, 1]`.
        '''
        sche = self.ddim_sche
        x_i = torch.randn(n_sample, *size).to(self.device)

        times = torch.arange(0, self.n_T, self.n_T // steps) + 1
        times = list(reversed(times.int().tolist())) + [0]
        time_pairs = list(zip(times[:-1], times[1:]))
        # e.g. [(801, 601), (601, 401), (401, 201), (201, 1), (1, 0)]

        for time, time_next in tqdm(time_pairs, disable=notqdm):
            t_is = torch.tensor([time / self.n_T]).to(self.device).repeat(n_sample)

            z = torch.randn(n_sample, *size).to(self.device) if time_next > 0 else 0

            alpha = sche["alphabar_t"][time]
            eps, x0_t = self.pred_eps_(x_i, t_is, alpha, use_amp)
            alpha_next = sche["alphabar_t"][time_next]
            c1 = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c2 = (1 - alpha_next - c1 ** 2).sqrt()
            x_i = alpha_next.sqrt() * x0_t + c2 * eps + c1 * z

        return unnormalize_to_zero_to_one(x_i)

    def pred_eps_(self, x, t, alpha, clip_x=True):
        def pred_eps_from_x0(x0):
            return (x - x0 * alpha.sqrt()) / (1 - alpha).sqrt()

        def pred_x0_from_eps(eps):
            return (x - (1 - alpha).sqrt() * eps) / alpha.sqrt()

        # get prediction of x0
    
        with torch.no_grad():
            eps = self.nn_model(x, t).float()    
         
        denoised = pred_x0_from_eps(eps)

        # pixel-space clipping (optional)
        if clip_x:
            denoised = torch.clip(denoised, -1., 1.)
            eps = pred_eps_from_x0(denoised)
        return eps, denoised
    
    def pred_eps_cond(self, x, t, alpha, cond, clip_x=True):
        def pred_eps_from_x0(x0):
            return (x - x0 * alpha.sqrt()) / (1 - alpha).sqrt()

        def pred_x0_from_eps(eps):
            return (x - (1 - alpha).sqrt() * eps) / alpha.sqrt()

        # get prediction of x0
        # 这里需要传入的是真值!而不是去噪对象
        with torch.no_grad():
            eps = self.nn_model(x, t, cond).float()      ##### 真正sample的时候需要重新写这一部分
         
        denoised = pred_x0_from_eps(eps)

        # pixel-space clipping (optional)
        if clip_x:
            denoised = torch.clip(denoised, -1., 1.)
            eps = pred_eps_from_x0(denoised)
        return eps, denoised