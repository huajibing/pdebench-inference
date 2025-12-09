# PDEBench 专家模型推理说明

PKU 机器学习基础 课程项目

## 1. 任务

### 1.1 1D Diffusion-Sorption

**数据集**：[https://darus.uni-stuttgart.de/file.xhtml?fileId=133020&version=8.0](https://darus.uni-stuttgart.de/file.xhtml?fileId=133020&version=8.0)
- File name: 1D_diff-sorp_NA_NA.h5

**FNO模型**：
- File name: 1D_diff-sorp_NA_NA_FNO_small.pt

**PINN模型**：
- File name: 1D_diff-sorp_NA_NA_0001.h5_PINN.pt-15000.pt

**U-Net模型**：
- File name: 1D_diff-sorp_NA_NA_Unet_small-PF-10.pt

### 1.2 2D Darcy Flow

**数据集**：[https://darus.uni-stuttgart.de/file.xhtml?fileId=133219&version=8.0](https://darus.uni-stuttgart.de/file.xhtml?fileId=133219&version=8.0)
- File name: 2D_DarcyFlow_beta1.0_Train.hdf5

**FNO模型**：
- File name: 2D_DarcyFlow_beta1.0_Train_FNO_small.pt

**U-Net模型**：
- File name: 2D_DarcyFlow_beta1.0_Train_Unet_small.pt

### 1.3 1D Burgers' Equation

**数据集**：[https://darus.uni-stuttgart.de/file.xhtml?fileId=281365&version=8.0](https://darus.uni-stuttgart.de/file.xhtml?fileId=281365&version=8.0)
- File name: 1D_Burgers_Sols_Nu1.0.hdf5

**FNO模型**：
- File name: 1D_Burgers_Sols_Nu1.0_FNO_small.pt

**U-Net模型**：
- File name: 1D_Burgers_Sols_Nu1.0_Unet_small-PF-10.pt

以上所有模型使用[PDEBench](https://github.com/pdebench/PDEBench)项目进行重新训练，下载链接：[北大网盘](https://disk.pku.edu.cn/link/AAC8AC215DAADC47AF9EDE26F90A461684)
文件名：models.zip
有效期限：永久有效

## 2. 运行指南

### TL;DR

首先需要确保以下文件在当前目录（或通过 `--data-dir` 指定的目录）中：

**数据集文件：**
- `2D_DarcyFlow_beta1.0_Train.hdf5` — 2D Darcy Flow 数据
- `1D_Burgers_Sols_Nu1.0.hdf5` — 1D Burgers 数据
- `1D_diff-sorp_NA_NA.h5` — 1D Diffusion-Sorption 数据

**模型文件：**
- `2D_DarcyFlow_beta1.0_Train_Unet_small.pt`
- `2D_DarcyFlow_beta1.0_Train_FNO_small.pt`
- `1D_Burgers_Sols_Nu1.0_Unet_small-PF-10.pt`
- `1D_Burgers_Sols_Nu1.0_FNO_small.pt`
- `1D_diff-sorp_NA_NA_Unet_small-PF-10.pt`
- `1D_diff-sorp_NA_NA_FNO_small.pt`
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

若要指定模型/数据集，请参考 [批量推理指南](https://github.com/huajibing/pdebench-inference/blob/main/BATCH_INFERENCE_GUIDE.md)；若不需要指定数据集进行推理，请参考 [推理接口文档](https://github.com/huajibing/pdebench-inference/blob/main/INFERENCE_API.md)。