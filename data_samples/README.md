# PDEBench 数据样本

本目录包含各模型的完整输入输出数据样本，用于参考和测试。

## 文件列表

| 文件 | 任务 | 模型 | 输入形状 | 输出形状 |
|------|------|------|----------|----------|
| `darcy_unet2d_sample.txt` | 2D Darcy Flow | UNet2d | (64, 64) | (64, 64) |
| `darcy_fno2d_sample.txt` | 2D Darcy Flow | FNO2d | (64, 64) | (64, 64) |
| `burgers_unet1d_sample.txt` | 1D Burgers | UNet1d | (10, 256) | (256,) |
| `burgers_fno1d_sample.txt` | 1D Burgers | FNO1d | (256, 10) | (256,) |
| `diffsorp_unet1d_sample.txt` | 1D Diff-Sorp | UNet1d | (10, 1024) | (1024,) |
| `diffsorp_fno1d_sample.txt` | 1D Diff-Sorp | FNO1d | (1024, 10) | (1024,) |

## 文件内容

每个样本文件包含：
- 模型和任务说明
- 输入/输出格式规范
- 完整的输入数据（numpy 数组格式）
- 模型预测输出
- 真实标签（ground truth）
- MSE 误差

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

所有样本均从 PDEBench 官方数据集中提取：
- 2D Darcy Flow: `2D_DarcyFlow_beta1.0_Train.hdf5` 第0个样本
- 1D Burgers: `1D_Burgers_Sols_Nu1.0.hdf5` 第0个样本
- 1D Diff-Sorp: `1D_diff-sorp_NA_NA.h5` seed='0001'
