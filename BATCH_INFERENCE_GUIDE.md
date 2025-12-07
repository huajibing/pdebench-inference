# PDEBench 数据集批量推理使用指南

本指南介绍如何使用 `pdebench_inference.py` 进行批量推理，包括在完整数据集上运行所有模型并保存结果。

---

## 1. 快速开始

### 1.1 环境要求

```bash
# 必需依赖
pip install torch numpy h5py

# 可选依赖（用于进度条显示）
pip install tqdm
```

### 1.2 文件准备

确保以下文件在当前目录（或通过 `--data-dir` 指定的目录）中：

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
- `1D_diff-sorp_NA_NA_0001.h5_PINN.pt-15000.pt` (PINN 模型)

### 1.3 基本使用

```bash
# 查看可用模型和数据集
python pdebench_inference.py --list

# 运行所有模型的批量推理（使用全部数据）
python pdebench_inference.py --run-all

# 快速测试（仅使用100个样本）
python pdebench_inference.py --run-all --n-samples 100
```

---

## 2. 命令行参数详解

### 2.1 模式选择参数

| 参数 | 说明 |
|------|------|
| `--list` | 列出可用的模型和数据集 |
| `--run-all` | 运行批量推理 |
| `--checkpoint <path>` | 加载单个模型进行测试 |

### 2.2 批量推理参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--tasks` | 全部 | 指定要评估的任务：`darcy`, `burgers`, `diffsorp` |
| `--models` | 全部 | 指定要使用的模型文件 |
| `--n-samples` | 全部 | 评估的样本数量 |
| `--batch-size` | 32 | 批处理大小 |
| `--output` | `inference_results` | 输出目录 |
| `--data-dir` | `.` | 数据和模型所在目录 |
| `--device` | 自动 | 运行设备：`cpu` 或 `cuda` |
| `--no-save-predictions` | False | 仅保存指标，不保存预测数组 |

### 2.3 使用示例

```bash
# 仅评估 Darcy Flow 任务
python pdebench_inference.py --run-all --tasks darcy

# 评估多个任务
python pdebench_inference.py --run-all --tasks darcy burgers

# 使用前500个样本，批大小64
python pdebench_inference.py --run-all --n-samples 500 --batch-size 64

# 指定输出目录
python pdebench_inference.py --run-all --output my_results/

# 仅使用特定模型
python pdebench_inference.py --run-all --models 2D_DarcyFlow_beta1.0_Train_FNO.pt

# 强制使用 CPU
python pdebench_inference.py --run-all --device cpu

# 仅保存指标（节省磁盘空间）
python pdebench_inference.py --run-all --no-save-predictions

# 数据在其他目录
python pdebench_inference.py --run-all --data-dir /path/to/data/
```

---

## 3. 支持的任务和模型

### 3.1 2D Darcy Flow

| 项目 | 说明 |
|------|------|
| 数据集 | `2D_DarcyFlow_beta1.0_Train.hdf5` |
| 样本数 | 10,000 |
| 输入 | 渗透率场 (64×64) |
| 输出 | 压力场 (64×64) |
| 模型 | UNet2d, FNO2d |

**预处理**：原始数据 128×128，下采样 2 倍至 64×64

### 3.2 1D Burgers

| 项目 | 说明 |
|------|------|
| 数据集 | `1D_Burgers_Sols_Nu1.0.hdf5` |
| 样本数 | 10,000 |
| 输入 | 前10个时间步的速度场 (10×256) |
| 输出 | 下一时间步的速度场 (256,) |
| 模型 | UNet1d, FNO1d |

**预处理**：
- 空间下采样 4 倍：1024 → 256
- 时间下采样 5 倍：201 → 41 时间步

### 3.3 1D Diffusion-Sorption

| 项目 | 说明 |
|------|------|
| 数据集 | `1D_diff-sorp_NA_NA.h5` |
| 样本数 | 10,000 |
| 输入 | 前10个时间步的浓度场 (10×1024) |
| 输出 | 下一时间步的浓度场 (1024,) |
| 模型 | UNet1d, FNO1d, **PINN1d** |

**预处理**：无下采样

**关于 PINN 模型**：
- PINN（Physics-Informed Neural Network）使用不同的推理方式
- 输入为 (x, t) 坐标而非时间序列数据
- 可直接预测任意空间-时间点的浓度值
- 脚本会自动检测 PINN 模型并使用正确的推理方式

---

## 4. 输出文件说明

### 4.1 目录结构

```
inference_results/
├── summary.json              # 所有模型的汇总指标
├── darcy/
│   ├── 2D_DarcyFlow_beta1.0_Train_Unet_PF_1_metrics.json
│   ├── 2D_DarcyFlow_beta1.0_Train_Unet_PF_1_predictions.npy
│   ├── 2D_DarcyFlow_beta1.0_Train_Unet_PF_1_targets.npy
│   ├── 2D_DarcyFlow_beta1.0_Train_FNO_metrics.json
│   ├── 2D_DarcyFlow_beta1.0_Train_FNO_predictions.npy
│   └── 2D_DarcyFlow_beta1.0_Train_FNO_targets.npy
├── burgers/
│   └── ...
└── diffsorp/
    └── ...
```

### 4.2 文件格式

**summary.json** — 汇总指标：
```json
{
  "2D_DarcyFlow_beta1.0_Train_Unet_PF_1": {
    "mse": 0.000335,
    "mae": 0.011042,
    "n_samples": 10000
  },
  "2D_DarcyFlow_beta1.0_Train_FNO": {
    "mse": 0.000945,
    "mae": 0.020745,
    "n_samples": 10000
  }
}
```

**{model}_metrics.json** — 单个模型的详细指标：
```json
{
  "model_name": "2D_DarcyFlow_beta1.0_Train_Unet_PF_1",
  "task_name": "Darcy",
  "n_samples": 10000,
  "mse": 0.000335,
  "mae": 0.011042,
  "timestamp": "2025-12-07T21:30:12.798347"
}
```

**{model}_predictions.npy** — 模型预测结果：
- Darcy: shape `(N, 64, 64)`
- Burgers: shape `(N, 256)`
- Diff-Sorp: shape `(N, 1024)`

**{model}_targets.npy** — 真实标签，shape 与 predictions 相同

### 4.3 加载结果

```python
import numpy as np
import json

# 加载指标
with open("inference_results/summary.json") as f:
    summary = json.load(f)

# 加载预测和标签
predictions = np.load("inference_results/darcy/2D_DarcyFlow_beta1.0_Train_Unet_PF_1_predictions.npy")
targets = np.load("inference_results/darcy/2D_DarcyFlow_beta1.0_Train_Unet_PF_1_targets.npy")

print(f"Predictions shape: {predictions.shape}")
print(f"MSE: {np.mean((predictions - targets) ** 2):.6e}")
```

---

## 5. 高级用法

### 5.1 仅评估部分样本范围

如果需要评估特定范围的样本（例如用于交叉验证），可以在 Python API 中指定：

```python
from pdebench_inference import PDEPredictor, DarcyDataset, run_batch_inference

# 加载数据集和模型
dataset = DarcyDataset("2D_DarcyFlow_beta1.0_Train.hdf5")
predictor = PDEPredictor.load("2D_DarcyFlow_beta1.0_Train_Unet_PF_1.pt")

# 仅评估样本 1000-2000
sample_indices = list(range(1000, 2000))
result = run_batch_inference(
    predictor,
    dataset,
    sample_indices=sample_indices,
    batch_size=32,
)

print(f"MSE: {result.mse:.6e}")
dataset.close()
```

### 5.2 自定义进度显示

如果没有安装 tqdm，脚本会使用简单的文本进度显示。建议安装 tqdm 获得更好的体验：

```bash
pip install tqdm
```

### 5.3 内存优化

对于大数据集，可以：

1. **不保存预测数组**：使用 `--no-save-predictions` 减少内存使用
2. **减小批大小**：使用 `--batch-size 16` 或更小的值
3. **分批处理**：使用 `--n-samples` 分多次运行

### 5.4 GPU 加速

如果有 CUDA GPU：

```bash
# 自动检测并使用 GPU
python pdebench_inference.py --run-all

# 强制使用 GPU
python pdebench_inference.py --run-all --device cuda

# 强制使用 CPU（调试用）
python pdebench_inference.py --run-all --device cpu
```

---

## 6. Python API 调用

### 6.1 基本批量推理

```python
from pdebench_inference import run_all_models

# 运行所有模型
results = run_all_models(
    data_dir=".",
    output_dir="my_results",
    n_samples=1000,
    batch_size=32,
)

# 访问结果
for model_name, result in results.items():
    print(f"{model_name}: MSE={result.mse:.6e}, MAE={result.mae:.6e}")
```

### 6.2 单任务评估

```python
from pdebench_inference import (
    PDEPredictor,
    DarcyDataset,
    BurgersDataset,
    DiffSorpDataset,
    run_batch_inference,
    save_results,
)

# 加载 Darcy 数据集
dataset = DarcyDataset(
    "2D_DarcyFlow_beta1.0_Train.hdf5",
    reduced_resolution=2,  # 下采样因子
)

# 加载模型
predictor = PDEPredictor.load("2D_DarcyFlow_beta1.0_Train_FNO.pt")

# 运行推理
result = run_batch_inference(
    predictor,
    dataset,
    sample_indices=list(range(100)),  # 前100个样本
    batch_size=32,
    save_predictions=True,
    show_progress=True,
)

# 保存结果
save_results(result, "output/", "darcy_fno")

# 清理
dataset.close()
```

### 6.3 PINN 模型推理

PINN 模型使用不同的推理方式，接受 (x, t) 坐标作为输入：

```python
from pdebench_inference import (
    PDEPredictor,
    DiffSorpDataset,
    run_pinn_inference,
)
import numpy as np

# 加载 PINN 模型
predictor = PDEPredictor.load("1D_diff-sorp_NA_NA_0001.h5_PINN.pt-15000.pt")
print(predictor)
# PDEPredictor(
#   model_type=PINN1d,
#   input_shape=[N, 2] where N is number of (x, t) query points,
#   out_channels=1,
#   device=cpu
# )

# 方式1：直接预测任意 (x, t) 点
x = np.linspace(0, 1, 100)        # 空间坐标
t = np.full(100, 250.0)           # 时间 t=250s
coords = np.stack([x, t], axis=-1).astype(np.float32)
output = predictor.predict(coords)  # shape: (100, 1)

# 方式2：在数据集上进行评估
dataset = DiffSorpDataset("1D_diff-sorp_NA_NA.h5", initial_step=10)

# 使用专门的 PINN 推理函数
result = run_pinn_inference(
    predictor,
    dataset,
    sample_indices=list(range(100)),
    save_predictions=True,
    show_progress=True,
)

print(f"MSE: {result.mse:.6e}")
print(f"MAE: {result.mae:.6e}")

dataset.close()
```

### 6.4 自定义数据处理

```python
from pdebench_inference import PDEPredictor, DarcyDataset
import numpy as np

# 加载数据集
dataset = DarcyDataset("2D_DarcyFlow_beta1.0_Train.hdf5")

# 获取单个样本
input_data, target = dataset[0]
print(f"Input shape: {input_data.shape}")  # (64, 64)
print(f"Target shape: {target.shape}")     # (64, 64)

# 获取批量样本
inputs, targets = dataset.get_batch([0, 1, 2, 3])
print(f"Batch inputs shape: {inputs.shape}")   # (4, 64, 64)
print(f"Batch targets shape: {targets.shape}") # (4, 64, 64)

# 准备模型输入
predictor = PDEPredictor.load("2D_DarcyFlow_beta1.0_Train_Unet_PF_1.pt")
model_input = dataset.prepare_input_unet(inputs)  # (4, 1, 64, 64)

# 推理
predictions = predictor.predict(model_input)
print(f"Predictions shape: {predictions.shape}")  # (4, 1, 64, 64)

dataset.close()
```