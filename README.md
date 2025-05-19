# 🌠 Sparse Diffusion Autoencoder for Test-time Adapting Prediction of Spatiotemporal Dynamics

![Demo](assets/comparison.gif)

Thank you for reviewing our **Neurips 2025** manuscript: 📄 *"Sparse Diffusion Autoencoder for Test-time Adapting Prediction of Spatiotemporal Dynamics"*

This repository contains the implementation of **SparseDiff**, a novel Sparse Diffusion Autoencoder that efficiently predicts spatiotemporal dynamics and dynamically self-adjusts at test-time.

![image](assets/SparseDiff.png)




## 🚀 Quick Start

### 📦 Required Dependencies

Ensure the following packages are installed before running the code:

```
pip install tqdm yaml torch torchdiffeq ema_pytorch torch_geometric torchmetrics
```





### 🏃 Running the Model (Example: SH System)

1️⃣ **Download Dataset** 📂: [Google Drive](https://drive.google.com/drive/folders/1i2A_Bw3mUXcsInx8DvZOaOT7vO57-p9L?usp=sharing)

Shape of the data:  (num_trajectories, steps, channel, x_dim, y_dim)

- uv.npy: (100, 100, 1, 128, 128)

- uv_test.npy: (50, 100, 1, 128, 128)
  
2️⃣ **Download model chekpoint** 📂: [Google Drive](https://drive.google.com/drive/folders/1i2A_Bw3mUXcsInx8DvZOaOT7vO57-p9L?usp=sharing)

We have three models: Sparse Encoder, Diffusive Predictor and Unconditioned diffusion.

Please download ***grand_input_10_256_1.pth*** (the predictor model weights) and ***vqvae_T_10_ae_pretrain_30_32_32.pth*** (the sparse encoder model weights) to the ./log/sh directory. Download ***model_999.pth*** (the diffusion model weights) to the ./log/sh/ckpts_55 directory.

3️⃣ **Run the model** :

After downloading the data and model weights, you can directly run ***sample_sh.ipynb*** to get the iterative prediction result for SH system! ✅


4️⃣ **Train the model**:

If you want to re-train the model, you can run train_sh.py with commands:

- For Single GPU / CPU:

  ```sh
  python run.py
  ```

- For Multi-GPU (with AMP support):

  ```sh
  python -m torch.distributed.run --master_port=25640 --nproc_per_node=8 train.py --use_amp --multi_gpu --system sh
  ```





## 📁 Repository Structure

```sh
.
├── README.md
├── config
│   └── SH.yaml
├── datasets.py
├── model
│   ├── __init__.py
│   ├── block.py
│   ├── common.py
│   ├── DDPM.py
│   ├── grand_predictor.py
│   ├── unet.py
│   └── vq_vae.py
├── log
│   ├── sh
│   │   ├── grand_input_10_256_1.pth
│   │   ├── vqvae_T_10_ae_pretrain_30_32_32.pth
│   │   ├── ckpts_55
│   │   │   └── model_999.pth
├── train_sh.py
├── sample_sh.ipynb
├── datasets.py
└── utils.py
```





## 📌 Notes

- This implementation provides a **demo using the SH system** as an example. (Full version will be released after acceptance)
- Supports **both single-GPU and multi-GPU training**.
- Configuration files are stored in the `config/` directory.
- For questions regarding reproducibility or additional details, please refer to our manuscript.
