# PDEBench 推理接口文档

本文档介绍 `pdebench_inference.py` 脚本支持的三个 PDE 任务：1D Burgers 方程、1D 扩散-吸附方程和 2D Darcy 流。每个任务包含物理背景、输入输出格式说明和真实数据示例。

---

## 快速开始

```python
from pdebench_inference import PDEPredictor

# 加载模型
predictor = PDEPredictor.load("checkpoint.pt")

# 查看模型信息和期望输入格式
print(predictor)

# 推理
output = predictor.predict(input_data)
```

---

## 1. 2D Darcy Flow（达西流）

### 1.1 物理背景

Darcy 流描述流体在多孔介质（如土壤、岩石）中的渗流运动。该问题求解稳态压力场。

**控制方程**（椭圆型 PDE）：
```
-∇·(a(x,y) ∇u(x,y)) = f(x,y)
```

其中：
- `a(x,y)` — 渗透率场（permeability），描述介质允许流体通过的能力
- `u(x,y)` — 压力场（pressure），即我们要求解的目标
- `f(x,y)` — 源项（source term）

**物理意义**：给定一个不均匀的多孔介质（渗透率场），预测其中的压力分布。渗透率高的区域流体更容易通过，压力梯度较小。

### 1.2 输入输出格式

| 项目 | 格式 | 说明 |
|------|------|------|
| **输入** | `[B, 1, Nx, Ny]` 或 `[B, Nx, Ny, 1]` | 渗透率场 `a(x,y)` |
| **输出** | `[B, 1, Nx, Ny]` 或 `[B, Nx, Ny, 1, 1]` | 压力场 `u(x,y)` |
| **空间范围** | `x, y ∈ [0, 1]` | 单位正方形区域 |
| **数值范围** | 渗透率: `[0.1, 1.0]`; 压力: `[0, 0.5]` | — |

**UNet2d 输入格式**：`[B, C, Nx, Ny]`（PyTorch 标准格式）

**FNO2d 输入格式**：`[B, Nx, Ny, C]`（通道在最后）

### 1.3 数据示例

```python
# 渗透率场（输入）- 8x8 采样示例
# 值为 0.1 表示低渗透率区域，值为 1.0 表示高渗透率区域
nu = np.array([
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    [0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 0.1, 0.1],  # 高渗透率区域开始
    [0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 0.1],
    [0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1],
    [0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 0.1, 0.1],
    [0.1, 0.1, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1],
])

# 压力场（输出）- 对应的 8x8 采样
# 压力从边界向中心逐渐增加
pressure = np.array([
    [0.002, 0.015, 0.020, 0.023, 0.023, 0.022, 0.020, 0.014],
    [0.015, 0.182, 0.266, 0.304, 0.313, 0.301, 0.262, 0.174],
    [0.020, 0.265, 0.397, 0.450, 0.457, 0.441, 0.390, 0.255],
    [0.022, 0.298, 0.440, 0.476, 0.460, 0.456, 0.443, 0.294],
    [0.023, 0.304, 0.425, 0.439, 0.446, 0.446, 0.438, 0.308],
    [0.023, 0.304, 0.400, 0.414, 0.422, 0.426, 0.423, 0.297],
    [0.021, 0.287, 0.369, 0.377, 0.385, 0.395, 0.391, 0.256],
    [0.015, 0.201, 0.327, 0.321, 0.329, 0.349, 0.265, 0.171],
])
```

### 1.4 使用示例

```python
from pdebench_inference import PDEPredictor
import numpy as np

# 加载模型
predictor = PDEPredictor.load("2D_DarcyFlow_beta1.0_Train_Unet_PF_1.pt")

# 创建输入：渗透率场 [B=1, C=1, Nx=64, Ny=64]
permeability = np.random.uniform(0.1, 1.0, (1, 1, 64, 64)).astype(np.float32)

# 预测压力场
pressure = predictor.predict(permeability)
# pressure.shape = (1, 1, 64, 64)
```

---

## 2. 1D Burgers 方程

### 2.1 物理背景

Burgers 方程是流体力学中描述激波形成和传播的经典方程，结合了非线性对流和粘性扩散效应。

**控制方程**（抛物型 PDE）：
```
∂u/∂t + u · ∂u/∂x = ν · ∂²u/∂x²
```

其中：
- `u(x,t)` — 流体速度场
- `ν` — 运动粘度（kinematic viscosity），本数据集中 `ν = 1.0`
- 左边第二项 `u · ∂u/∂x` — 非线性对流项，导致激波形成
- 右边项 — 粘性扩散项，使激波平滑

**物理意义**：给定初始速度分布，预测流体速度随时间的演化。非线性项会使速度剖面变陡形成激波，而粘性会平滑激波。

### 2.2 输入输出格式

| 项目 | 格式 | 说明 |
|------|------|------|
| **输入** | 前 `T₀` 个时间步的速度场 | 作为初始条件 |
| **输出** | 下一个时间步的速度场 | 自回归预测 |
| **空间范围** | `x ∈ [0, 1]`（周期边界） | — |
| **时间范围** | `t ∈ [0, 2]` | 201 个时间步 |
| **数值范围** | 速度: `[0.71, 0.83]` | — |

**UNet1d 输入格式**：`[B, T₀, Nx]`，其中 `T₀=10`（前10个时间步）

**FNO1d 输入格式**：`[B, Nx, T₀]`（时间步在最后一维）

### 2.3 数据示例

```python
# 速度场随时间演化 - 采样示例
# shape: [T=5, Nx=8]，每 128 个空间点采样一次
# 可以看到初始的双峰分布逐渐趋于平滑

velocity = np.array([
    # t=0: 初始状态，明显的高低速度交替（类似正弦波）
    [0.832, 0.715, 0.832, 0.715, 0.832, 0.715, 0.832, 0.715],
    # t=1: 开始扩散
    [0.782, 0.766, 0.781, 0.766, 0.781, 0.766, 0.781, 0.766],
    # t=2: 继续平滑
    [0.775, 0.773, 0.775, 0.773, 0.775, 0.773, 0.775, 0.773],
    # t=3: 接近均匀
    [0.774, 0.773, 0.774, 0.773, 0.774, 0.773, 0.774, 0.773],
    # t=4: 趋于稳态
    [0.774, 0.773, 0.774, 0.773, 0.774, 0.773, 0.774, 0.773],
])

# 物理解释：
# - 初始时刻有明显的速度梯度 (0.832 vs 0.715)
# - 粘性作用使速度梯度逐渐减小
# - 最终趋向于均匀分布
```

### 2.4 使用示例

```python
from pdebench_inference import PDEPredictor
import numpy as np

# 加载模型
predictor = PDEPredictor.load("1D_Burgers_Sols_Nu1.0_Unet-PF-20.pt")

# 创建输入：前10个时间步的速度场 [B=1, T=10, Nx=256]
initial_velocity = np.random.uniform(0.7, 0.85, (1, 10, 256)).astype(np.float32)

# 预测下一个时间步
next_velocity = predictor.predict(initial_velocity)
# next_velocity.shape = (1, 1, 256)

# 自回归预测多个时间步
predictions = predictor.predict_autoregressive(initial_velocity, num_steps=20)
# predictions.shape = (1, 256, 20, 1)
```

---

## 3. 1D Diffusion-Sorption（扩散-吸附方程）

### 3.1 物理背景

扩散-吸附方程描述溶质在多孔介质中的传输过程，考虑了分子扩散和固体表面吸附效应。广泛应用于地下水污染物传输、土壤修复等环境工程问题。

**控制方程**（反应-扩散型 PDE）：
```
∂u/∂t = D · ∂²u/∂x² - R(u)
```

其中：
- `u(x,t)` — 溶质浓度
- `D = 5×10⁻⁴` — 扩散系数
- `R(u)` — 吸附反应项（Freundlich 吸附等温线）

**边界条件**：
- 左边界 (`x=0`)：恒定浓度 `u = 1`（污染源注入）
- 右边界 (`x=1`)：零通量（无出流）

**物理意义**：模拟污染物从左侧注入后，如何通过扩散和吸附作用在介质中传播。

### 3.2 输入输出格式

| 项目 | 格式 | 说明 |
|------|------|------|
| **输入** | 前 `T₀` 个时间步的浓度场 | 作为初始条件 |
| **输出** | 下一个时间步的浓度场 | 自回归预测 |
| **空间范围** | `x ∈ [0, 1]` | — |
| **时间范围** | `t ∈ [0, 500]` | 101 个时间步 |
| **数值范围** | 浓度: `[0, 1]` | 归一化浓度 |

**UNet1d 输入格式**：`[B, T₀, Nx]`，其中 `T₀=10`

**FNO1d 输入格式**：`[B, Nx, T₀]`

### 3.3 数据示例

```python
# 浓度场随时间演化 - 采样示例
# shape: [T=6, Nx=8]，每 20 个时间步采样一次，每 128 个空间点采样一次

concentration = np.array([
    # t=0: 初始均匀低浓度
    [0.083, 0.083, 0.083, 0.083, 0.083, 0.083, 0.083, 0.083],
    # t=100: 污染前沿开始推进（左侧浓度升高）
    [0.997, 0.506, 0.205, 0.103, 0.085, 0.082, 0.074, 0.048],
    # t=200: 继续扩散
    [0.998, 0.641, 0.353, 0.184, 0.110, 0.082, 0.063, 0.036],
    # t=300: 扩散加深
    [0.998, 0.706, 0.446, 0.261, 0.152, 0.095, 0.061, 0.031],
    # t=400: 接近稳态
    [0.999, 0.746, 0.508, 0.323, 0.196, 0.117, 0.067, 0.032],
    # t=500: 最终状态
    [0.999, 0.774, 0.553, 0.372, 0.236, 0.143, 0.080, 0.036],
])

# 物理解释：
# - 左边界 (x=0) 始终保持高浓度 ≈ 1.0（污染源）
# - 浓度梯度从左向右递减
# - 随时间推移，污染前沿逐渐向右推进
# - 吸附作用减缓了传播速度
```

### 3.4 使用示例

```python
from pdebench_inference import PDEPredictor
import numpy as np

# 加载模型
predictor = PDEPredictor.load("1D_diff-sorp_NA_NA_Unet-PF-20.pt")

# 创建输入：前10个时间步的浓度场 [B=1, T=10, Nx=1024]
initial_concentration = np.zeros((1, 10, 1024), dtype=np.float32)
initial_concentration[:, :, 0] = 1.0  # 左边界恒定浓度

# 预测下一个时间步
next_concentration = predictor.predict(initial_concentration)
# next_concentration.shape = (1, 1, 1024)

# 自回归预测多个时间步
predictions = predictor.predict_autoregressive(initial_concentration, num_steps=50)
# predictions.shape = (1, 1024, 50, 1)
```

---

## 4. 可用模型总结

| 任务 | 模型 | Checkpoint 文件 | 输入格式 | 预训练参数 |
|------|------|----------------|----------|-----------|
| 2D Darcy Flow | UNet2d | `2D_DarcyFlow_beta1.0_Train_Unet_PF_1.pt` | `[B, 1, 64, 64]` | reduced_resolution=2 |
| 2D Darcy Flow | FNO2d | `2D_DarcyFlow_beta1.0_Train_FNO.pt` | `[B, 64, 64, 1]` | modes=12, width=20 |
| 1D Burgers | UNet1d | `1D_Burgers_Sols_Nu1.0_Unet-PF-20.pt` | `[B, 10, 256]` | initial_step=10 |
| 1D Burgers | FNO1d | `1D_Burgers_Sols_Nu1.0_FNO.pt` | `[B, 256, 10]` | modes=12, width=20 |
| 1D Diff-Sorp | UNet1d | `1D_diff-sorp_NA_NA_Unet-PF-20.pt` | `[B, 10, 1024]` | initial_step=10 |
| 1D Diff-Sorp | FNO1d | `1D_diff-sorp_NA_NA_FNO.pt` | `[B, 1024, 10]` | modes=16, width=64 |

---

## 5. 完整调用示例

```python
from pdebench_inference import PDEPredictor, list_available_models
import numpy as np

# 列出可用模型
models = list_available_models()
for m in models:
    print(f"{m['name']}: {m['model_type']}")

# 加载任意模型
predictor = PDEPredictor.load("2D_DarcyFlow_beta1.0_Train_Unet_PF_1.pt")

# 查看期望输入格式
print(predictor)
# PDEPredictor(
#   model_type=UNet2d,
#   input_shape=[B, 1, Nx, Ny],
#   out_channels=1,
#   device=cpu
# )

# 构造输入数据
permeability = np.random.uniform(0.1, 1.0, (1, 1, 64, 64)).astype(np.float32)

# 推理
pressure = predictor.predict(permeability)
print(f"Output shape: {pressure.shape}")  # (1, 1, 64, 64)
```

---

## 6. 数据预处理注意事项

### 6.1 空间分辨率下采样

预训练模型使用了下采样的数据进行训练：

| 任务 | 原始分辨率 | 训练分辨率 | 下采样因子 |
|------|-----------|-----------|-----------|
| 2D Darcy Flow | 128×128 | 64×64 | 2 |
| 1D Burgers | 1024 | 256 | 4 |
| 1D Diff-Sorp | 1024 | 1024 | 1 |

```python
# 下采样示例
def downsample_1d(data, factor):
    return data[..., ::factor]

def downsample_2d(data, factor):
    return data[..., ::factor, ::factor]
```

### 6.2 时间分辨率下采样（用于时序任务）

| 任务 | 原始时间步 | 训练时间步 | 下采样因子 |
|------|-----------|-----------|-----------|
| 1D Burgers | 201 | 41 | 5 |
| 1D Diff-Sorp | 101 | 101 | 1 |

---

## 7. API 参考

### `PDEPredictor.load(checkpoint_path, device=None)`

从 checkpoint 文件加载模型。

**参数**：
- `checkpoint_path`: str 或 Path，checkpoint 文件路径
- `device`: 可选，运行设备（默认自动选择 CUDA/CPU）

**返回**：`PDEPredictor` 实例

### `predictor.predict(input_data, grid=None)`

单步推理。

**参数**：
- `input_data`: numpy 数组或 torch 张量
- `grid`: 可选，FNO 模型的网格坐标（默认自动生成）

**返回**：numpy 数组

### `predictor.predict_autoregressive(initial_condition, num_steps, grid=None)`

自回归多步预测。

**参数**：
- `initial_condition`: 初始条件
- `num_steps`: 预测步数
- `grid`: 可选

**返回**：所有时间步的预测结果

### `list_available_models(directory=".")`

扫描目录中的可用模型。

**返回**：模型信息字典列表

---

## 8. 依赖要求

```
torch>=1.9.0
numpy>=1.19.0
```

脚本为自包含设计，无需安装 PDEBench 或其他额外依赖。
