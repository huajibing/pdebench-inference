# PDEBench 专家模型跑通说明

## 1. 任务

### 1.1 1D Diffusion-Sorption

**数据集**：[https://darus.uni-stuttgart.de/file.xhtml?fileId=133020&version=8.0](https://darus.uni-stuttgart.de/file.xhtml?fileId=133020&version=8.0)
- File name: 1D_diff-sorp_NA_NA.h5

**FNO模型**：[https://darus.uni-stuttgart.de/file.xhtml?persistentId=doi:10.18419/DARUS-2987/5&version=2.0](https://darus.uni-stuttgart.de/file.xhtml?persistentId=doi:10.18419/DARUS-2987/5&version=2.0)
- File name: 1D_diff-sorp_NA_NA_FNO.pt

**PINN模型**：[https://darus.uni-stuttgart.de/file.xhtml?persistentId=doi:10.18419/DARUS-2987/2&version=2.0](https://darus.uni-stuttgart.de/file.xhtml?persistentId=doi:10.18419/DARUS-2987/2&version=2.0)
- File name: 1D_diff-sorp_NA_NA_0001.h5_PINN.pt-15000.pt

**U-Net模型**：[https://darus.uni-stuttgart.de/file.xhtml?persistentId=doi:10.18419/DARUS-2987/6&version=2.0](https://darus.uni-stuttgart.de/file.xhtml?persistentId=doi:10.18419/DARUS-2987/6&version=2.0)
- File name: 1D_diff-sorp_NA_NA_Unet-PF-20.pt

### 1.2 2D Darcy Flow

**数据集**：[https://darus.uni-stuttgart.de/file.xhtml?fileId=133219&version=8.0](https://darus.uni-stuttgart.de/file.xhtml?fileId=133219&version=8.0)
- File name: 2D_DarcyFlow_beta1.0_Train.hdf5

**FNO模型**：[https://darus.uni-stuttgart.de/file.xhtml?persistentId=doi:10.18419/DARUS-2987/27&version=2.0](https://darus.uni-stuttgart.de/file.xhtml?persistentId=doi:10.18419/DARUS-2987/27&version=2.0)
- File name: DarcyFlow_FNO.tar, 解压后选择 2D_DarcyFlow_beta1.0_Train_FNO.pt

**U-Net模型**：[https://darus.uni-stuttgart.de/file.xhtml?persistentId=doi:10.18419/DARUS-2987/34&version=2.0](https://darus.uni-stuttgart.de/file.xhtml?persistentId=doi:10.18419/DARUS-2987/34&version=2.0)
- File name: DarcyFlow_Unet.tar, 解压后选择 2D_DarcyFlow_beta1.0_Train_Unet_PF_1.pt

### 1.3 1D Burgers' Equation

**数据集**：[https://darus.uni-stuttgart.de/file.xhtml?fileId=281365&version=8.0](https://darus.uni-stuttgart.de/file.xhtml?fileId=281365&version=8.0)
- File name: 1D_Burgers_Sols_Nu1.0.hdf5

**FNO模型**：[https://darus.uni-stuttgart.de/file.xhtml?persistentId=doi:10.18419/DARUS-2987/26&version=2.0](https://darus.uni-stuttgart.de/file.xhtml?persistentId=doi:10.18419/DARUS-2987/26&version=2.0)
- File name: burgers_FNO.tar, 解压后选择 1D_Burgers_Sols_Nu1.0_FNO.pt

**U-Net模型**：[https://darus.uni-stuttgart.de/file.xhtml?persistentId=doi:10.18419/DARUS-2987/33&version=2.0](https://darus.uni-stuttgart.de/file.xhtml?persistentId=doi:10.18419/DARUS-2987/33&version=2.0)
- File name: burgers_Unet.tar, 解压后选择 1D_Burgers_Sols_Nu1.0_Unet-PF-20.pt

## 2. 运行指南

### TL;DR

首先需要确保以下文件在当前目录（或通过 `--data-dir` 指定的目录）中：

**数据集文件：**
- `2D_DarcyFlow_beta1.0_Train.hdf5` — 2D Darcy Flow 数据
- `1D_Burgers_Sols_Nu1.0.hdf5` — 1D Burgers 数据
- `1D_diff-sorp_NA_NA.h5` — 1D Diffusion-Sorption 数据

**模型文件：**
- `2D_DarcyFlow_beta1.0_Train_Unet_PF_1.pt`
- `2D_DarcyFlow_beta1.0_Train_FNO.pt`
- `1D_Burgers_Sols_Nu1.0_Unet-PF-20.pt`
- `1D_Burgers_Sols_Nu1.0_FNO.pt`
- `1D_diff-sorp_NA_NA_Unet-PF-20.pt`
- `1D_diff-sorp_NA_NA_FNO.pt`
- `1D_diff-sorp_NA_NA_0001.h5_PINN.pt-15000.pt`

安装依赖：
```python
pip install torch numpy h5py # 必需依赖
pip install tqdm # 可选，用于进度条显示
```

跑所有模型的推理：
```python
python pdebench_inference.py --run-all
```

默认设置下会在 `inference_results/` 中生成推理结果和评估指标。

> **测试环境**：NVIDIA RTX 4090 (24GB), Python 3.9, Ubuntu 22.04, CUDA 12.8, PyTorch 2.8.0
> **运行时间**：约 25s

### 更多功能

若要指定模型/数据集，请参考 [批量推理指南](BATCH_INFERENCE_GUIDE.md)；若不需要指定数据集进行推理，请参考 [推理接口文档](INFERENCE_API.md)。