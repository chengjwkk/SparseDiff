# 🌊 MDPNet: Multiscale Diffusion Autoencoder for Complex Systems

Thank you for reviewing our **KDD 2025** manuscript: 📄 *"Predicting the Dynamics of Complex Systems via Multiscale Diffusion Autoencoder"*

This repository contains the implementation of **MDPNet**, a novel approach for learning and predicting the evolution of complex dynamical systems using multiscale diffusion-based autoencoders.

![image](assets/SparseDiff.pdf)




## 🚀 Quick Start

### 📦 Required Dependencies

Ensure the following packages are installed before running the code:

```
pip install tqdm yaml torch torchdiffeq ema_pytorch
```





### 🏃 Running the Model (Example: GS System)

1️⃣ **Download Dataset** 📂: [Google Drive](https://drive.google.com/file/d/17-GSDZN4olVQaqBDbHzQq-UAhlwfGHoh/view?usp=sharing)

2️⃣ **Run the model**:

- Single GPU / CPU:

  ```sh
  python run.py
  ```

- Multi-GPU (with AMP support):

  ```sh
  python -m torch.distributed.run --master_port=25640 --nproc_per_node=8 train.py --use_amp --multi_gpu
  ```





## 📁 Repository Structure

```sh
.
├── README.md
├── config
│   └── GS.yaml
├── datasets.py
├── model
│   ├── __init__.py
│   ├── cEncoder.py
│   ├── cPredictor.py
│   ├── cUnet.py
│   ├── common.py
│   └── mDynaDDPM.py
├── run.py
└── utils.py
```





## 📌 Notes

- This implementation provides a **demo using the GS system** as an example. (Full version will be released after acceptance)
- Supports **both single-GPU and multi-GPU training**.
- Configuration files are stored in the `config/` directory.
- For questions regarding reproducibility or additional details, please refer to our manuscript.
