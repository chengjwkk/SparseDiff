import torch
import gc

def generate_sparse_observations(encoding_indices, data, M=30):
    '''
        data: (B, 128*128, 10), 应该是过去的时间点
    '''
    
    B, _ = encoding_indices.shape  # (B, 128*128)
    H, W = 128, 128
    input_steps = 10

    # Initialize X, Y, and Coord for all batches
    X = torch.zeros((B * M, input_steps))  # (B*M, 10)
   
    Coord = torch.full((B * M, 2), -1.0)  # (B*M, 2), coordinates initialized to -1

    

    # Iterate over each batch
    for b in range(B):
        indices = encoding_indices[b]  # (128*128)
        batch_data = data[b]  # (128*128, 10)
      
        unique_codes = torch.unique(indices)  # Get unique codes for the current batch

        # Process each unique code (at most M points)
        for code in unique_codes:
            mask = indices == code
            selected_idx = torch.nonzero(mask).squeeze()
            if selected_idx.dim() == 0:
                selected_idx = selected_idx.unsqueeze(0)

            # Randomly choose one index
            chosen_idx = selected_idx[torch.randint(0, len(selected_idx), (1,))]

            # Get (x, y) coordinates normalized by image dimensions
            y, x = divmod(chosen_idx.item(), W)
            y = y / float(W)
            x = x / float(H)
            index_counter = b*M + code
            # Update Coord with normalized coordinates
            Coord[index_counter, :] = torch.tensor([x, y])

            # Update X with corresponding data
            X[index_counter, :] = batch_data[chosen_idx, :].squeeze()
          


    # Unfilled parts will remain as -1 in Coord, and 0 in X
    return X, Coord   # (B*M, 10) ; (B*M, 2)

import torch
import numpy as np

def generate_edge_index_2(encoding_indices, M):
    """
    Generate edge_index and edge_weight for sparse observation points based on codebook adjacency.
    Now edge_index uses actual codebook indices (0~99), and includes all adjacent code pairs,
    even if their co-occurrence weight is zero.
    
    Returns:
        edge_index: (2, num_edges) torch tensor with codebook indices in [0, codebook_size)
        edge_weight: (num_edges, 1) torch tensor of normalized weights (may include 0s)
    """
    H, W = 128, 128
    encoding_grid = encoding_indices.view(H, W).numpy()
    codebook_size = M  # 显式写出codebook大小

    cooccur = np.zeros((codebook_size, codebook_size), dtype=np.float32)
    neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1),
                        (0, -1),          (0, 1),
                        (1, -1),  (1, 0), (1, 1)]

    for i in range(H):
        for j in range(W):
            current_code = encoding_grid[i, j]
            for dx, dy in neighbor_offsets:
                ni, nj = i + dx, j + dy
                if 0 <= ni < H and 0 <= nj < W:
                    neighbor_code = encoding_grid[ni, nj]
                    if neighbor_code != current_code:
                        cooccur[current_code, neighbor_code] += 1

    np.fill_diagonal(cooccur, 0)

    # Normalize
    non_zero_counts = (cooccur > 0).sum(axis=1, keepdims=True)
    norm_cooccur = cooccur / (non_zero_counts + 1e-8)

    edge_list = []
    edge_weights = []

    for i in range(codebook_size):
        related_scores = norm_cooccur[i]
        for j in range(codebook_size):
            if i != j:    ##全连接
                edge_list.append((i, j))
                edge_weights.append(related_scores[j])  # 可能为0

    edge_index = torch.tensor(edge_list, dtype=torch.long).T  # (2, num_edges)
    edge_weight = torch.tensor(edge_weights, dtype=torch.float32)

    # Normalize to [0, 1] if not all 0
    if edge_weight.max() > edge_weight.min():
        edge_weight = (edge_weight - edge_weight.min()) / (edge_weight.max() - edge_weight.min())
    edge_weight = edge_weight.unsqueeze(-1)

    return edge_index, edge_weight

def generate_edge_index(encoding_indices, M):
    """
    Generate edge_index and edge_weight for sparse observation points based on codebook adjacency.
    Now edge_index uses actual codebook indices (0~99), and includes all adjacent code pairs,
    even if their co-occurrence weight is zero. The function is adapted for batched inputs.
    
    Returns:
        edge_index: (2, num_edges * B) torch tensor with codebook indices in [0, codebook_size)
        edge_weight: (num_edges * B, 1) torch tensor of normalized weights (may include 0s)
    """
    B, _ = encoding_indices.shape  # (B, 128*128)
    H, W = 128, 128
    codebook_size = M  # Explicit codebook size
    
    # Initialize the lists to collect all edges and their corresponding weights
    edge_list = []
    edge_weights = []

    # Iterate over each batch
    for b in range(B):
        encoding_grid = encoding_indices[b].view(H, W).numpy()

        cooccur = np.zeros((codebook_size, codebook_size), dtype=np.float32)
        neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1),
                            (0, -1),          (0, 1),
                            (1, -1),  (1, 0), (1, 1)]

        # Count co-occurrence for the current batch
        for i in range(H):
            for j in range(W):
                current_code = encoding_grid[i, j]
                for dx, dy in neighbor_offsets:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < H and 0 <= nj < W:
                        neighbor_code = encoding_grid[ni, nj]
                        if neighbor_code != current_code:
                            cooccur[current_code, neighbor_code] += 1

        np.fill_diagonal(cooccur, 0)

        # Normalize co-occurrence matrix
        non_zero_counts = (cooccur > 0).sum(axis=1, keepdims=True)
        norm_cooccur = cooccur / (non_zero_counts + 1e-8)

        # Append edges and weights for the current batch
        for i in range(codebook_size):
            related_scores = norm_cooccur[i]
            for j in range(codebook_size):
                if i != j:  # Full connection
                    edge_list.append((i, j))
                    edge_weights.append(related_scores[j])  # Could be zero

    # Convert lists to tensors
    edge_index = torch.tensor(edge_list, dtype=torch.long).T  # (2, num_edges * B)
    edge_weight = torch.tensor(edge_weights, dtype=torch.float32)  # (num_edges * B,)

    # Normalize edge_weight to [0, 1] if not all 0
    if edge_weight.max() > edge_weight.min():
        edge_weight = (edge_weight - edge_weight.min()) / (edge_weight.max() - edge_weight.min())
    edge_weight = edge_weight.unsqueeze(-1)  # (num_edges * B, 1)

    return edge_index, edge_weight


import os
import yaml
import torch
import matplotlib.pyplot as plt
from model import DDPM, UNet_new, VQVAE, GraphModel
from ema_pytorch import EMA
from utils import Config
import numpy as np
import random
import torch.nn.functional as F
from setproctitle import setproctitle
from torch_geometric.data import Data

import random
import argparse

def normalize_to_neg_one_to_one(img):
    # [0.0, 1.0] -> [-1.0, 1.0]
    return img * 2 - 1
def unnormalize_to_zero_to_one(t):
    # [-1.0, 1.0] -> [0.0, 1.0]
    return (t + 1) * 0.5

# 用full引导，就不需要square_sparse了？（不过可以画图）

def reconstruct_from_sparse(Sparse, encoding_indices, n ,grid_size=128,  strategy='first'):   
    # 之后需要输入每一个sparse对应的坐标，对于坐标点，保持为真值；非坐标点，用坐标点的均值代替 （可以先全部赋值，然后对应点换成真值）
    # 返回两个mask和mask-，mask-的obs要小一些，mask+大一些。
    # pde相同
    
    """
    将稀疏观测值映射回完整图像（支持每个码元多个观测值）。

    参数：
    - Sparse: (B, C, M*n)，每个码元对应 n 个观测值
    - encoding_indices: (B, C, H*W)，每个像素的码元编号，值 ∈ [0, M-1]
    - grid_size: 图像边长
    - n: 每个码元的观测点数量
    - strategy: 'first'（默认）或 'mean'

    返回：
    - reconstructed: (B, C, H, W)
    """
    B, C, total_M = Sparse.shape
    H = W = grid_size

    assert encoding_indices.shape == (B, C, H * W)
   

    # 先 reshape： (B, C, M*n) → (B, C, M, n)
    Sparse_grouped = Sparse.view(B, C, -1, n)

    if strategy == 'first':
        Sparse_selected = Sparse_grouped[:, :, :, 0]  # shape: (B, C, M)
    elif strategy == 'mean':
        Sparse_selected = Sparse_grouped.mean(dim=3)  # shape: (B, C, M)
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

   

    # Gather 稀疏值（按 M 维度）
    reconstructed_flat = torch.gather(Sparse_selected, dim=2, index=encoding_indices)  # shape: (B, C, H*W)

    # reshape 成图像
    reconstructed = reconstructed_flat.view(B, C, H, W)
    return reconstructed

def reconstruct_from_sparse_batch(Sparse, encoding_indices, sample_step, n ,grid_size=128,  strategy='first'):   
    # 之后需要输入每一个sparse对应的坐标，对于坐标点，保持为真值；非坐标点，用坐标点的均值代替 （可以先全部赋值，然后对应点换成真值）
    # 返回两个mask和mask-，mask-的obs要小一些，mask+大一些。
    # pde相同
    
    """
    将稀疏观测值映射回完整图像（支持每个码元多个观测值）。

    参数：
    - Sparse: (B*sample_step, C, M*n)，每个码元对应 n 个观测值
    - encoding_indices: (B, C, H*W)，每个像素的码元编号，值 ∈ [0, M-1]
    - grid_size: 图像边长
    - n: 每个码元的观测点数量
    - strategy: 'first'（默认）或 'mean'

    返回：
    - reconstructed: (B, C, H, W)
    """
    B_1, C, total_M = Sparse.shape
    H = W = grid_size
    B = int(B_1 / sample_step)

   
    
    # 先 reshape： (B*sample_step, C, M*n) → (B, C, M, n)
    Sparse_grouped = Sparse.view(B*sample_step, C, -1, n)

    if strategy == 'first':
        Sparse_selected = Sparse_grouped[:, :, :, 0]  # shape: (B*sample_step, C, M)
    elif strategy == 'mean':
        Sparse_selected = Sparse_grouped.mean(dim=3)  # shape: (B*sample_step, C, M)
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    print(encoding_indices.shape)
    encoding_indices = encoding_indices.unsqueeze(1).expand(-1, sample_step, -1, -1).reshape(B*sample_step, C, -1)
    print("1", encoding_indices.shape)
    print(Sparse_selected.shape)
    # Gather 稀疏值（按 M 维度）
    reconstructed_flat = torch.gather(Sparse_selected, dim=2, index=encoding_indices)  # shape: (B*sample_step, C, H*W)

    # reshape 成图像
    reconstructed = reconstructed_flat.view(B_1, C, H, W)
    return reconstructed



def generate_sample_sparse_observations(n_sample, encoding_indices, data, M, n=1):
    """
    随机采样稀疏观测点，保证每个码元有 n 个观测（不足时重复），
    映射规则为：code → Sparse[:, c, code*n : (code+1)*n]

    Args:
        n_sample: int, 时间步数量
        start_t: int, 起始时间步
        encoding_indices: (C, H*W)，每个像素对应的编码（整数）
        data: (C, H*W, batch_size)，原始数据
        M: 每个 channel 最多观测码元数（每个码元选 n 个）
        n: 每个码元选取的观测点数

    Returns:
        Sparse: (n_sample, C, M*n)
        Coord: (C, M*n, 2)
    """
    C, HW = encoding_indices.shape
    H = W = int(HW ** 0.5)
    T = data.shape[2]

    total_M = M * n
    Sparse = torch.zeros((n_sample, C, total_M))
    Coord = torch.full((C, total_M, 2), -1.0)

    for c in range(C):
        indices = encoding_indices[c]  # shape: (H*W,)
        unique_codes = torch.unique(indices)

        for code in unique_codes:
            code_int = int(code.item())

            mask = (indices == code)
            selected_idx = torch.nonzero(mask).squeeze()
            if selected_idx.dim() == 0:
                selected_idx = selected_idx.unsqueeze(0)

            num_available = selected_idx.shape[0]

            target_start = code_int * n

            if num_available < n:
                print(f"[Warning] Code {code_int} in channel {c} only has {num_available} points, replicating to fill {n}.")
                # 先填入所有可用的点
                for i in range(num_available):
                    idx = selected_idx[i]
                    x, y = divmod(idx.item(), W)
                    Coord[c, target_start + i] = torch.tensor([x, y], dtype=torch.float32)
                    Sparse[:, c, target_start + i] = data[c, idx]
                # 随机重复采样填满
                if num_available > 0:
                    repeat_needed = n - num_available
                    repeated = selected_idx[torch.randint(0, num_available, (repeat_needed,))]
                    for i, idx in enumerate(repeated):
                        x, y = divmod(idx.item(), W)
                        Coord[c, target_start + num_available + i] = torch.tensor([x, y], dtype=torch.float32)
                        Sparse[:, c, target_start + num_available + i] = data[c, idx]
            else:
                chosen_indices = selected_idx[torch.randperm(num_available)[:n]]
                for i, idx in enumerate(chosen_indices):
                    x, y = divmod(idx.item(), W)
                    Coord[c, target_start + i] = torch.tensor([x, y], dtype=torch.float32)
                    Sparse[:, c, target_start + i] = data[c, idx]

    return Sparse, Coord

def generate_mask_and_sparse(Coord, Sparse, grid_size=128):
    """
    生成掩码矩阵和稀疏观测值矩阵。

    参数：
    Coord: 3D tensor (channels, M*n, 2)，包含稀疏点坐标。
    Sparse: 3D tensor (batch_size, channels, M*n)，每行包含对应批次的稀疏观测点值。
    grid_size: 网格大小，默认为128。
    device: 使用的设备（CUDA或CPU）。

    返回：
    mask: channels x 128 x 128大小的掩码矩阵，标记有效的观测点。
    square_sparse: batch_size x channels x 128 x 128大小的稀疏矩阵，对应稀疏点的值。
    """
    # 恢复到真实坐标（直接乘以grid_size）
    x_coords = Coord[:,:,0].long()  # (channels , N)
    y_coords = Coord[:,:,1].long()  # 

    # 初始化mask和square_sparse
    mask = torch.zeros((Coord.size(0), grid_size, grid_size), dtype=torch.float32)
    square_sparse = torch.zeros((Sparse.size(0), Sparse.size(1), grid_size, grid_size), dtype=torch.float32)  # batch_size x channels x 128 x 128

    # 遍历所有批次
    for batch_idx in range(Sparse.size(0)):  # 遍历batch_size
        # 遍历所有通道
        for channel_idx in range(Sparse.size(1)):  # 遍历channels
            # 遍历所有坐标点，忽略坐标为 -1 的点
            for i in range(len(Sparse[batch_idx, channel_idx])):
                # 如果坐标为 -1，认为是无效点，跳过
                if x_coords[channel_idx, i] == -1 or y_coords[channel_idx, i] == -1:
                    continue

                # 获取对应的x、y坐标索引
                x_idx = x_coords[channel_idx, i].item()
                y_idx = y_coords[channel_idx, i].item()

                # 确保坐标在有效范围内
                if 0 <= x_idx < grid_size and 0 <= y_idx < grid_size:
                    mask[channel_idx, x_idx, y_idx] = 1  # 标记为观测点
                    square_sparse[batch_idx, channel_idx, x_idx, y_idx] = Sparse[batch_idx, channel_idx, i]  # 填充观测值
    
    
    return mask, square_sparse

def main():

    # ==== 1. 设置 config 路径并读取 yaml ====
    system = 'sh'
    yaml_path = f'config/{system}.yaml'
    setproctitle("@chengjingwen")

    with open(yaml_path, 'r') as f:
        base_opt = yaml.full_load(f)  # 读取成 dict


    # ==== 2. 读取命令行参数 ====
    parser = argparse.ArgumentParser()
    parser.add_argument('--zeta_obs', type=float, default=None)
    parser.add_argument('--zeta_pde', type=float, default=None)
    parser.add_argument('--ratio', type=float, default=None)
    parser.add_argument('--gpu', type=int, default=None)
    args, _ = parser.parse_known_args() 
    
    # ==== 3. 用命令行参数覆盖 config 中的字段 ====
    if "sample" not in base_opt:
        base_opt["sample"] = {}

    if args.zeta_obs is not None:
        base_opt["sample"]["zeta_obs"] = args.zeta_obs
    if args.zeta_pde is not None:
        base_opt["sample"]["zeta_pde"] = args.zeta_pde
    if args.ratio is not None:
        base_opt["sample"]["ratio"] = args.ratio
    if args.gpu is not None:
        base_opt["gpu"] = args.gpu
    

    # # 设置 GPU（默认 GPU 0）
    # base_opt["gpu"] = args.gpu if args.gpu is not None else 1

    # ==== 4. 构建 Config 对象并设置 device ====

    opt = Config(base_opt)
    # device = torch.device(f"cuda:{opt.gpu}")
    
    device = torch.device(f"cuda:{opt.gpu}")

    # 设置保存路径
    opt.save_dir += f'/{opt.dataset}/'
    model_dir = os.path.join(opt.save_dir, "ckpts_55")
    sample_dir = os.path.join(opt.save_dir, "sample_55")
    os.makedirs(sample_dir, exist_ok=True)

    # 初始化模型
    diff = DDPM(
        nn_model=UNet_new(**opt.network),
        **opt.diffusion,
        device=device,
    )
    diff.to(device)
    
    # 加载模型权重
    load_epoch = 999
    checkpoint_path = os.path.join(model_dir, f"model_{load_epoch}.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    diff.load_state_dict(checkpoint['MODEL'], strict=False)

    # 初始化并加载 EMA
    ema = EMA(diff, beta=opt.ema, update_after_step=0, update_every=1)
    ema.to(device)
    ema.load_state_dict(checkpoint['EMA'], strict=False)

    # ===================
    # 使用 ema 模型采样
    # ===================
    ema_model = ema.ema_model
    ema_model.eval()

    # 设置采样参数
    size = (opt.data["channel"], opt.data['img_resolution'], opt.data['img_resolution'])  # C, H, W，假设是灰度图像

    # ===================
    # load vq-vae获取码元
    # ===================
    T= opt.vqvae["T"]
    hidden_dim = opt.vqvae["hidden_dim"]    # mlp的隐藏层维度
    embedding_dim = opt.vqvae["embedding_dim"]    # 隐空间维度
    num_embeddings =  opt.vqvae["num_embeddings"]
    vqvae = VQVAE(input_dim=T, hidden_dim=hidden_dim, embedding_dim=embedding_dim, num_embeddings=num_embeddings)
    vqvae.load_state_dict(torch.load(f"/data3/chengjingwen/diffusion-mdpnet/trained_model/{system}/vqvae_T_{T}_ae_pretrain_{num_embeddings}_{hidden_dim}_{embedding_dim}.pth"))
    vqvae.eval()

    # ===================
    # load predictor
    # ===================
    latent_feat = opt.grand["latent_feat"]
    input_steps = opt.grand["input_steps"]

    predictor = GraphModel(input_steps=input_steps, d_in = latent_feat, codebook_size=num_embeddings).to(device)
    predictor.load_state_dict(torch.load(f"/data3/chengjingwen/diffusion-mdpnet/trained_model/{system}/grand_input_{T}_{latent_feat}_1.pth"))

    # ===================
    # load data
    # ===================
    x_test = np.load(f'/data3/chengjingwen/diffusion-mdpnet/data/{system}/uv_test.npy')  # Shape: (num_tra, steps, channels, 128, 128)
    x_test = torch.tensor(x_test, dtype=torch.float32)  

    x_train = np.load(f'/data3/chengjingwen/diffusion-mdpnet/data/{system}/uv.npy')  # Shape: (num_tra, steps, channels, 128, 128)
    x_train = torch.tensor(x_train, dtype=torch.float32)  

    # Normalize data (Min-Max Scaling)
    xmin = x_train.amin(dim=(0, 1, 3, 4), keepdim=True)  # Min over num_tra, steps, spatial dimensions
    xmax = x_train.amax(dim=(0, 1, 3, 4), keepdim=True)  # Max over num_tra, steps, spatial dimensions
    del x_train
    test_data = (x_test - xmin) / (xmax - xmin)  # Normalized data

    batch_size = 1   # 最好可以整除test_tra (50)   #### 检查一下，为什么batch_size>1会报错越界
    sample_step = 5  # 必须可以整除10  <=10
    channel = opt.data["channel"]
    # n = opt.n_per_area
    n = 1

    # test_tra = opt.sample["num_tra"]
    test_tra = 1
    # pred_steps = opt.sample["steps"] - 1
    pred_steps = 80    # >= 10
    start_step = 2*T
    t_input = torch.tensor([1.0])
    change_step = 5

    # TODO:
    # 需要可视化任何一条轨迹的话，就在外部定义好trajectory，获取truth，声明predictions。

    truth = test_data[:test_tra , start_step:start_step + pred_steps, :,:,:]   # (batch_size, T, 2, 128, 128)
    predictions = torch.zeros_like(truth)

    with torch.no_grad():
        for i in range(0, test_tra, batch_size):    
            count = 0
            for code_step in range(start_step, start_step + pred_steps, change_step):  # 在这一层更新encoding_indices(码元)
                if(code_step == start_step):
                    for_codebook_data = test_data[i : i + batch_size, code_step-T:code_step, :,:,:]   # (B, T, 2, 128, 128)
                
                elif(code_step-start_step-T<0):  # 自回归
                    count += count
                    for_codebook_data = torch.cat([test_data[i : i + batch_size, code_step - T + change_step * count:code_step, :,:,:], predictions[i:i+batch_size, :change_step*count,:,:,:]],dim = 1)
                    ## TODO:
                else:
                    for_codebook_data = predictions[i:i+batch_size, code_step - start_step - T: code_step - start_step,:,:,:]

                single_tra_data_1 = for_codebook_data[:, :,0,:,:].reshape(batch_size, T, 128*128).permute(0,2,1)   # (B, 128*128, T)
                _, _, encoding_indices_1 = vqvae(single_tra_data_1)   # (B, 128*128)
               
                encoding_indices = encoding_indices_1.unsqueeze(1)
            
            
                # 有可能是剩下三个参数的问题，X没问题
                # 随机在每个区域选择一个稀疏观测点
                X_1, Coord_1 = generate_sparse_observations(encoding_indices_1, single_tra_data_1, num_embeddings)
                edge_index_1, edge_weight_1 = generate_edge_index(encoding_indices_1, num_embeddings)
             
                
                initial_data_1 = Data(    
                        x = X_1,   # (batch_size * M, T)
                        edge_index = edge_index_1,  # (2, E*batch_size)
                        edge_weight=edge_weight_1,  # (E*batch_size, 1)
                        Coord = Coord_1   #   (M*batch_size, 2)
                    )
              
                
                initial_data_1 = initial_data_1.to(device)
               
                
                
            
                sparse_predictions = []
                for step in range(0, change_step, sample_step):
                    
                    
                    if(step != 0):   # 此时用新sample出来的data重新生成X_1和Coord_1
                        ddpm_samples = ddpm_samples.reshape(batch_size*sample_step, channel, -1).reshape(batch_size, sample_step, channel, -1).detach().cpu()
                        ddpm_result_1 = ddpm_samples[:, :, 0,:].permute(0,2,1)
                       
                        # 去掉 single_tra_data_1 的前 pred_steps 步
                        
                        update_single_tra_data_1 = single_tra_data_1[:, :, sample_step:]
                        print(update_single_tra_data_1.shape)
                        update_single_tra_data_1 = torch.cat((update_single_tra_data_1, ddpm_result_1), dim=2)
                        
                       
                        
                        X_1, Coord_1 = generate_sparse_observations(encoding_indices_1, update_single_tra_data_1)  # (B, 128*128, T)

                        initial_data_1.x = X_1.to(device)
     
                        initial_data_1.Coord = Coord_1.to(device)
               
                    
                    
                    for t in range(sample_step):
                        # print(initial_data_1.x)
                        pred_1 = predictor(initial_data_1, t_input)  # 输入data整体，输出 [num_nodes*batch_size, 1]
                        sparse_predictions.append(pred_1.detach().cpu())
            

                        initial_data_1.x = initial_data_1.x[:, 1:]     # 更新最新步
                        initial_data_1.x = torch.cat((initial_data_1.x, pred_1), dim=1)
                        
                        
                      
                     
                    
                    ######### x和encoding_indices的生成没有问题。check 其余三个参数
                    
                    # x: (batch_size * M, T)  
                    ### 注意这里第一维的堆叠方式是 M, M, M，因此reshape时batch_size要先在第二维 
                    sparse_1 = initial_data_1.x[:,-sample_step:].detach().cpu().reshape(batch_size, -1, sample_step).permute(0,2,1).reshape(batch_size*sample_step, -1)
                   
                    Sparse = sparse_1.unsqueeze(1)   # (B*sample_step, C, M)
                    print(Sparse.shape)
                    
                    
                    
                    # Sparse, Coord = generate_sample_sparse_observations(sample_step*batch_size, encoding_indices, pred_sample.permute(2,3,0,1).reshape(128*128, batch_size*sample_step, 2).permute(2, 0, 1), M=100, n=n)
                    # mask, square_sparse = generate_mask_and_sparse(Coord, Sparse, grid_size=128)
                    reconstructed = reconstruct_from_sparse_batch(Sparse = Sparse, encoding_indices=encoding_indices, sample_step = sample_step, n = n)   #(B*sample_step, C, H, w)
                    
                    with torch.enable_grad():
                        # ddpm_samples = ema_model.ddim_guided_sample_full_sh(n_sample = batch_size*sample_step, size = size, steps = opt.sample["ddim_step"], eta = opt.sample["eta"], zeta_obs = opt.sample["zeta_obs"], zeta_pde = opt.sample["zeta_pde"], ratio = opt.sample["ratio"], reconstructed = reconstructed, data_opt = opt.data, notqdm=False)  # shape: (B*sample_step, C, H, W)
                        ddpm_samples = ema_model.ddim_sample_from_reconstructed_sh(n_sample = batch_size*sample_step, size = size, steps = 50, eta = opt.sample["eta"], zeta_obs = opt.sample["zeta_obs"], zeta_pde = opt.sample["zeta_pde"], ratio = opt.sample["ratio"], reconstructed = reconstructed, data_opt = opt.data, notqdm=False)  # shape: (B*sample_step, C, H, W)
                        predictions[i:i+batch_size, code_step - start_step + step:code_step - start_step + step + sample_step] = ddpm_samples.reshape(batch_size, sample_step, channel, 128, 128).detach().cpu()
                print(F.mse_loss(predictions, truth, reduction='none').mean().numpy())
                    
            del initial_data_1
            del ddpm_samples, reconstructed, sparse_predictions
            torch.cuda.empty_cache()
                

        # channel = 0
        # truth = sample_data[start_t:start_t+n_sample, channel, :, :]  
        # print("train min:", truth.min().item(), "max:", truth.max().item(), "mean:", truth.mean().item())
        # reconstructed = reconstruct_from_sparse(Sparse = Sparse, encoding_indices=encoding_indices, n=n)


        # print("Sample min:", samples.min().item(), "max:", samples.max().item(), "mean:", samples.mean().item())
        # # 可视化采样结果

        # samples = samples[:, channel].detach().cpu() 

        # mse_list = F.mse_loss(samples, truth, reduction='none')  # (n_sample, 128, 128)
        # mse_values = mse_list.mean(dim=(1, 2)).numpy()

        # mean_mse = mse_values.mean()
        # square_sparse = unnormalize_to_zero_to_one(square_sparse)
        # print(f"[INFO] MEAN MSE: {mean_mse:.4e}")
        # plt.figure(figsize=(2.5 * n_sample, 10))

        # # 第一行：Ground Truth
        # for i in range(n_sample):
        #     plt.subplot(4, n_sample, i + 1)
        #     plt.imshow(truth[i].numpy(), cmap='coolwarm', vmin=0, vmax=0.8)
        #     plt.title("Ground Truth")
        #     plt.axis('off')

        # # 第二行：Samples + MSE
        # for i in range(n_sample):
        #     plt.subplot(4, n_sample, n_sample + i + 1)
        #     plt.imshow(samples[i].numpy(), cmap='coolwarm', vmin=0, vmax=0.8)
        #     plt.title(f"Sample\nMSE: {mse_values[i]:.4e}")
        #     plt.axis('off')

        # # 第三行：Square Sparse（观测点）
        # for i in range(n_sample):
        #     plt.subplot(4, n_sample, 2 * n_sample + i + 1)
        #     plt.imshow(square_sparse[i, channel].numpy(), cmap='coolwarm', vmin=0, vmax=0.8)
        #     plt.title("Sparse Obs")
        #     plt.axis('off')

        # # 第四行：Constructed（由稀疏点扩展得到）+ MSE
        # constructed_mse_values = F.mse_loss(reconstructed[:, channel], truth, reduction='none')
        # constructed_mse_values = constructed_mse_values.view(n_sample, -1).mean(dim=1)  # 每个样本的 MSE


        # for i in range(n_sample):
        #     plt.subplot(4, n_sample, 3 * n_sample + i + 1)
        #     plt.imshow(reconstructed[i, channel].numpy(), cmap='coolwarm', vmin=0, vmax=0.8)
        #     plt.title(f"Constructed\nMSE: {constructed_mse_values[i]:.4e}")
        #     plt.axis('off')

        # plt.tight_layout()
        # filename = f"{n}_{opt.sample['zeta_obs']}_{opt.sample['zeta_pde']}_{opt.sample['ratio']}_{opt.sample['proportion']}.png"
        # plt.savefig(os.path.join(sample_dir, filename), bbox_inches='tight', dpi=300)
        # print(f"采样结果保存在 {sample_dir}/{filename}")

        from torchmetrics.functional import structural_similarity_index_measure as ssim


        def compute_nmse(predictions, targets):
            """计算 NMSE，每一步返回一个指标"""
            # 对每个时间步 (pred_steps) 计算 NMSE，沿着 (num_var, x_dim, y_dim) 维度求均值
        
            nmse = torch.mean((predictions - targets) ** 2, dim=(0, 2, 3, 4)) / torch.mean((targets) ** 2, dim=(0, 2, 3, 4))
            return nmse  # 返回一个形状为 (pred_steps,) 的张量

        def compute_rmse(predictions, targets):
            """计算 RMSE，每一步返回一个指标"""
            rmse = torch.sqrt(torch.mean((predictions - targets) ** 2, dim=(0, 2, 3, 4)))
            return rmse  # 返回一个形状为 (pred_steps,) 的张量

        def compute_ssim(predictions, targets):
            """计算 SSIM，每一步返回一个指标"""
            ssim_values = []
            for t in range(predictions.shape[1]):  # predictions.shape[1] 是时间步数，这里是 pred_steps
                ssim_value = ssim(predictions[:,t], targets[:,t], data_range=1.0).item()
                ssim_values.append(ssim_value)
            return torch.tensor(ssim_values)

        # 现在的结果是有问题的，check一下和.ipynb有什么区别吧
        # predictions 和 truth的 形状: (batch_size, T, 2, 128, 128)
        nmse = compute_nmse(predictions, truth)    # (pred_steps, )
        my_ssim = compute_ssim(predictions, truth)
        rmse = compute_rmse(predictions, truth)

        print(f"nmse: {nmse.mean()};{nmse.std()}\n; ssim: {my_ssim.mean()};{my_ssim.std()}\n, rmse: {rmse.mean()};{rmse.std()}")
        
        del diff, ema, ema_model, predictor, vqvae, test_data, x_test, predictions, truth
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()
    
    