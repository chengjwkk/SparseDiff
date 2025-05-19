import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize  # 使用官方库

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)  # 新增一层 Linear
        self.fc4 = nn.Linear(hidden_dim, embedding_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.norm4 = nn.LayerNorm(embedding_dim)
        
    def forward(self, x):
        x = F.gelu(self.norm1(self.fc1(x)))
        x = F.gelu(self.norm2(self.fc2(x)))
        x = F.gelu(self.norm3(self.fc3(x)))  # 额外的 Linear 层
        x = self.norm4(self.fc4(x))  # 输出层
        return x  # (B, embedding_dim)


class MLPDecoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)  # 新增一层 Linear
        self.fc4 = nn.Linear(hidden_dim, output_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = F.gelu(self.norm1(self.fc1(x)))
        x = F.gelu(self.norm2(self.fc2(x)))
        x = F.gelu(self.norm3(self.fc3(x)))  # 额外的 Linear 层
        x = self.fc4(x)  # 输出层
        return x  # (B, output_dim)

class VQVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_embeddings):
        super().__init__()
        self.encoder = MLPEncoder(input_dim, hidden_dim, embedding_dim)
        self.vq = VectorQuantize(
            dim=embedding_dim, 
            codebook_size=num_embeddings, 
            decay=0.6,  # 使用 EMA 更新码表
            kmeans_init = True,
            use_cosine_sim = False,   # 这可以让码本loss和重建loss在一个数量级
            commitment_weight=1.0,
            rotation_trick = False    # False is recommended
        )
        self.decoder = MLPDecoder(embedding_dim, hidden_dim, input_dim)

    def forward(self, x):
        encoded = self.encoder(x)  # (B, x*y, embedding_dim)
        quantized, indices, commit_loss = self.vq(encoded)  # (B, x*y, embedding_dim)
        reconstructed = self.decoder(quantized)  # (B, x*y, T)

        return reconstructed, commit_loss, indices



