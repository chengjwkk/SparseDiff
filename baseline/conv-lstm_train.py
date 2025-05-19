
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import torch
import numpy as np
from setproctitle import setproctitle
setproctitle("@convlstm")

system = "sh"
x_train = np.load(f'/data3/chengjingwen/diffusion-mdpnet/data/{system}/uv.npy')
x_train = torch.tensor(x_train, dtype=torch.float32)  
print(x_train.shape)
num_trajectories, total_steps, num_var, x_dim, y_dim = x_train.shape

xmin = x_train.amin(dim=(0, 1, 3, 4), keepdim=True)  
xmax = x_train.amax(dim=(0, 1, 3, 4), keepdim=True)  


data = (x_train - xmin) / (xmax - xmin)


print(f"Training data normalized: min={data.min().item()}, max={data.max().item()}")


# torch.save({'xmin': xmin, 'xmax': xmax}, f'/data4/chengjingwen/kdd25-main 3/data/lo/normalization_params.pth')

num_train = int(num_trajectories * 0.8)  # 80%的轨迹数
random_indices = torch.randperm(num_trajectories)[:num_train]  
data = data[random_indices]
num_trajectories, total_steps, num_var, x_dim, y_dim = data.shape

import torch
from torch.utils.data import Dataset, DataLoader, random_split

class MultiStepInputDataset(Dataset):
    def __init__(self, data, history_steps=10):
        """
        data: Tensor, shape (num_trajectories, num_steps, num_variables, x_dim, y_dim)
        history_steps: how many past steps to use as input
        """
        self.data = data
        self.history_steps = history_steps
        self.num_trajectories, self.num_steps, self.num_variables, self.x_dim, self.y_dim = data.shape

        # 确保有足够时间步能取 history_steps + 1（因为input要10步，output是第11步）
        if self.num_steps <= self.history_steps:
            raise ValueError("num_steps must be greater than history_steps")

    def __len__(self):
        # 每条轨迹生成 (num_steps - history_steps) 个样本
        return self.num_trajectories * (self.num_steps - self.history_steps)

    def __getitem__(self, idx):
        # 确定轨迹编号和在轨迹内的起点时间步
        trajectory_idx = idx // (self.num_steps - self.history_steps)
        time_idx = idx % (self.num_steps - self.history_steps)

        # input: 连续 history_steps 个时间步，堆叠成channel
        input_data = self.data[trajectory_idx, time_idx : time_idx + self.history_steps]  # (history_steps, num_variables, x_dim, y_dim)
        

        # target: 预测的第 history_steps 后的一个时间步
        target_data = self.data[trajectory_idx, time_idx + self.history_steps].unsqueeze(0)  # (1, num_variables, x_dim, y_dim)

        return input_data, target_data


# train_data = data

# train_dataset = MultiStepInputDataset(train_data, history_steps=10)

# # 随机划分 80% train, 20% val
# train_size = int(0.8 * len(train_dataset))
# val_size = len(train_dataset) - train_size
# train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# # 打印一个batch看看
# for inputs, targets in train_loader:
#     print(f"Input shape: {inputs.shape}, Target shape: {targets.shape}")
#     num_variables = inputs.shape[1]
#     break  # 只打印一次

from convlstm import ConvLSTM
import torch
import torch.nn as nn
class my_ConvLSTM(nn.Module):
    """

    inputs: [B, time_step, num_var, x_dim, y_dim] or [B, num_var, x_dim, y_dim]
    outputs: [B, pred_steps, num_var, x_dim, y_dim]
    """
    def __init__(self, num_var, x_dim=128, y_dim=128, hidden_dim=128, num_layers=3, n=1):
        super(my_ConvLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.n = n  # Default prediction steps
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.num_var = num_var
        self.cell = ConvLSTM(
            input_dim = num_var, 
            hidden_dim = hidden_dim,
            kernel_size = (3,3),
            num_layers = 3,    # make sure len(hidden_dim) = num_layers
            batch_first=True, 
            bias=True, 
            return_all_layers=False)

        self.final_conv = nn.Conv2d(in_channels=hidden_dim, out_channels=num_var, kernel_size=3, padding = 1)

    def forward(self, x, n=None):
        if n is None:  # Use default n if not provided
            n = self.n
            
        if len(x.shape) == 4:   # add time dimension
            x = x.unsqueeze(1)
            
        # Pass through LSTM
        layer_output_list, last_state_list = self.cell(x)   # x: [batch_size, T, num_var, input_dim, input_dim]
        (h,c) = last_state_list[0]
        last_state_list = [(h, c)] * self.num_layers
        
        output = layer_output_list[0][:,-1,:,:]  # 最后一层layer、最后一个时间步的输出  output:[batch_size, hidden_dim, input_dim, input_dim]
        final_output = self.final_conv(output)  # [batch_size, num_var, input_dim, input_dim]
        y = [final_output]
        
        # Multi-step prediction
        for _ in range(1, n):
            last_output = y[-1].unsqueeze(1)  # [batch_size, 1, hidden_dim, input_dim, input_dim]
            
            layer_output_list, last_state_list = self.cell(last_output, last_state_list)   
            (h,c) = last_state_list[0]
            last_state_list = [(h, c)] * self.num_layers
            
            output = layer_output_list[0][:,-1,:,:]  #[batch_size, hidden_dim, input_dim, input_dim]
            final_output = self.final_conv(output)  # [batch_size, num_var, input_dim, input_dim]
            
            y.append(final_output)
        final_outputs = torch.stack(y, dim=1).reshape(-1, self.num_var, self.x_dim, self.y_dim)
        # y_tensor: [batch_size, n, hidden_dim, input_dim, input_dim]
       
        # Return last prediction and all predictions
        return final_outputs.reshape(-1, n, self.num_var, self.x_dim, self.y_dim)  # [batch_size, n, num_var, input_dim, input_dim]
    
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

x_dim = 128
y_dim = 128
input_step = 10
pred_step = 1
num_epochs = 15
batch_size = 8
device = "cuda:7" 
lr_min = 1e-5
hidden_dim = 128
num_layers = 3
train_data = data

train_dataset = MultiStepInputDataset(train_data, history_steps=10)

# 随机划分 80% train, 20% val
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 打印一个batch看看
for inputs, targets in train_loader:
    print(f"Input shape: {inputs.shape}, Target shape: {targets.shape}")
    num_variables = inputs.shape[1]
    break  # 只打印一次


Conv_LSTM = my_ConvLSTM(num_var, x_dim=x_dim, y_dim = y_dim, hidden_dim=hidden_dim, num_layers=num_layers, n=pred_step).to(device)

num_params = sum(p.numel() for p in Conv_LSTM.parameters())
print(f"模型总参数量: {num_params / 1e6:.2f} M")

optimizer = Adam(list(Conv_LSTM.parameters()), lr=1e-3)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True
)


criterion = nn.MSELoss()


for epoch in range(num_epochs):
    Conv_LSTM.train()
   
    total_loss = 0.0
    
    for batch_idx, (input_data, target_data) in tqdm(enumerate(train_loader), total = len(train_loader)):
        
        input_data = input_data.to(device)  # Shape: [batch_size, num_variables, x_dim, y_dim]
        target_data = target_data.to(device)  # Shape: [batch_size, pred_step, num_variables, x_dim, y_dim]
        
        
        pred_data = Conv_LSTM(input_data)
        
        
        loss = criterion(pred_data, target_data)
        total_loss += loss.item()

        
        optimizer.zero_grad()
        
        loss.backward()
        optimizer.step()
        
    avg_loss = total_loss / len(train_loader)
    scheduler.step(avg_loss)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Reconstruction Loss: {avg_loss:.2e}")
    print(f"learning rate for lstm:{optimizer.param_groups[0]['lr']}")

    if optimizer.param_groups[0]['lr'] < lr_min:
        print(f"Learning rate below threshold ({lr_min}). Stopping early.")
        break    
       

    

print("Training complete!")


torch.save(Conv_LSTM.state_dict(), f"/data3/chengjingwen/diffusion-mdpnet/trained_model/baseline/{system}/convlstm_{hidden_dim}_{num_layers}.pth")

print("Models saved successfully!")