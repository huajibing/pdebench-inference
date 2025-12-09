# PDEBench 数据样本

本目录包含各模型的完整输入输出数据样本，用于参考和测试。

**注意**：这些样本对应的模型与原始 PDEBench 预训练模型的维度不同。

## 文件列表

| 文件 | 任务 | 模型 | 输入形状 | 输出形状 |
|------|------|------|----------|----------|
| `darcy_unet2d_sample.txt` | 2D Darcy Flow | UNet2d | (32, 32) | (32, 32) |
| `darcy_fno2d_sample.txt` | 2D Darcy Flow | FNO2d | (32, 32) | (32, 32) |
| `burgers_unet1d_sample.txt` | 1D Burgers | UNet1d | (5, 64) | (64,) |
| `burgers_fno1d_sample.txt` | 1D Burgers | FNO1d | (64, 5) | (64,) |
| `diffsorp_unet1d_sample.txt` | 1D Diff-Sorp | UNet1d | (5, 64) | (64,) |
| `diffsorp_fno1d_sample.txt` | 1D Diff-Sorp | FNO1d | (64, 5) | (64,) |

## 维度对比

| 任务 | 原始维度 | 小维度 | 下采样因子 |
|------|----------|--------|-----------|
| 2D Darcy Flow | 128×128 | 32×32 | 4 (空间) |
| 1D Burgers | T₀=10, Nx=256 | T₀=5, Nx=64 | 16 (空间), 5 (时间) |
| 1D Diff-Sorp | T₀=10, Nx=1024 | T₀=5, Nx=64 | 16 (空间), 2 (时间) |

## 文件内容

每个样本文件包含：
- 模型和任务说明
- 输入/输出格式规范
- 完整的输入数据（numpy 数组格式）
- 真实标签（ground truth）

## 使用方式

```python
import numpy as np

# 可以直接复制文件中的数组定义来使用
# 或者用以下方式加载

# 示例：从文件中提取数据（需要自行解析）
with open('data_samples/darcy_unet2d_sample.txt', 'r') as f:
    content = f.read()
    # 解析 numpy 数组...
```

## 数据来源

所有样本均从 PDEBench 官方数据集中提取，经过下采样处理：
- 2D Darcy Flow: `2D_DarcyFlow_beta1.0_Train.hdf5` 第0个样本
- 1D Burgers: `1D_Burgers_Sols_Nu1.0.hdf5` 第0个样本
- 1D Diff-Sorp: `1D_diff-sorp_NA_NA.h5` seed='0001'

提取脚本：`extract_small_samples.py`
