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

> **完整数据样本**：见 `data_samples/darcy_unet2d_sample.txt` 和 `data_samples/darcy_fno2d_sample.txt`

以下为 8×8 采样（不是真实样本），用于展示数据的物理含义：

```python
# 渗透率场（输入）- 8x8 展示用采样
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
predictor = PDEPredictor.load("2D_DarcyFlow_beta1.0_Train_Unet_small.pt")

# 创建输入：渗透率场 [B=1, C=1, Nx=32, Ny=32]
permeability = np.random.uniform(0.1, 1.0, (1, 1, 32, 32)).astype(np.float32)

# 预测压力场
pressure = predictor.predict(permeability)
# pressure.shape = (1, 1, 32, 32)
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

**UNet1d 输入格式**：`[B, T₀, Nx]`，其中 `T₀=5`（前5个时间步），`Nx=64`

**FNO1d 输入格式**：`[B, Nx, T₀]`（时间步在最后一维），`Nx=64`，`T₀=5`

### 2.3 数据示例

> **完整数据样本**：见 `data_samples/burgers_unet1d_sample.txt` 和 `data_samples/burgers_fno1d_sample.txt`

以下为 Nx=8 采样（不是真实样本），用于展示数据的物理含义：

```python
# 速度场随时间演化 - 展示用采样
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
predictor = PDEPredictor.load("1D_Burgers_Sols_Nu1.0_Unet_small.pt")

# 创建输入：前5个时间步的速度场 [B=1, T=5, Nx=64]
initial_velocity = np.random.uniform(0.7, 0.85, (1, 5, 64)).astype(np.float32)

# 预测下一个时间步
next_velocity = predictor.predict(initial_velocity)
# next_velocity.shape = (1, 1, 64)

# 自回归预测多个时间步
predictions = predictor.predict_autoregressive(initial_velocity, num_steps=20)
# predictions.shape = (1, 20, 64)
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

**UNet1d 输入格式**：`[B, T₀, Nx]`，其中 `T₀=5`，`Nx=64`

**FNO1d 输入格式**：`[B, Nx, T₀]`，`Nx=64`，`T₀=5`

### 3.3 数据示例

> **完整数据样本**：见 `data_samples/diffsorp_unet1d_sample.txt` 和 `data_samples/diffsorp_fno1d_sample.txt`

以下为 Nx=8 采样（不是真实样本），用于展示数据的物理含义：

```python
# 浓度场随时间演化 - 展示用采样
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

### 3.4 使用示例（UNet/FNO）

```python
from pdebench_inference import PDEPredictor
import numpy as np

# 加载模型
predictor = PDEPredictor.load("1D_diff-sorp_NA_NA_Unet_small.pt")

# 创建输入：前5个时间步的浓度场 [B=1, T=5, Nx=64]
initial_concentration = np.zeros((1, 5, 64), dtype=np.float32)
initial_concentration[:, :, 0] = 1.0  # 左边界恒定浓度

# 预测下一个时间步
next_concentration = predictor.predict(initial_concentration)
# next_concentration.shape = (1, 1, 64)

# 自回归预测多个时间步
predictions = predictor.predict_autoregressive(initial_concentration, num_steps=30)
# predictions.shape = (1, 30, 64)
```

### 3.5 PINN 模型

**PINN（Physics-Informed Neural Network）** 使用完全不同的推理方式：

| 项目 | 说明 |
|------|------|
| **输入** | `[N, 2]` — (x, t) 坐标点 |
| **输出** | `[N, 1]` — 每个点的浓度值 |
| **优势** | 可直接预测任意时空点，无需自回归迭代 |
| **架构** | 全连接网络 [2] → [40]×6 → [1]，tanh 激活 |

**PINN 使用示例**：

```python
from pdebench_inference import PDEPredictor
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

# 直接预测任意 (x, t) 点的浓度
# 例如：预测 t=250s 时刻，x 从 0 到 1 的浓度分布
x = np.linspace(0, 1, 1024)
t = np.full(1024, 250.0)  # t=250 秒
coords = np.stack([x, t], axis=-1).astype(np.float32)

concentration = predictor.predict(coords)
# concentration.shape = (1024, 1)

# 预测整个时空域（例如绘制时空演化图）
x = np.linspace(0, 1, 100)
t = np.linspace(0, 500, 50)
xx, tt = np.meshgrid(x, t)
coords = np.stack([xx.flatten(), tt.flatten()], axis=-1).astype(np.float32)

concentration = predictor.predict(coords).reshape(50, 100)
# concentration.shape = (50, 100) — 时间 × 空间
```

---

## 4. 可用模型总结

| 任务 | 模型 | Checkpoint 文件 | 输入格式 | 输出格式 | 预训练参数 |
|------|------|----------------|----------|-----------|-------|
| 2D Darcy Flow | UNet2d | `2D_DarcyFlow_beta1.0_Train_Unet_small.pt` | `[B, 1, 32, 32]` | `[B, 1, 32, 32]` | reduced_resolution=4 |
| 2D Darcy Flow | FNO2d | `2D_DarcyFlow_beta1.0_Train_FNO_small.pt` | `[B, 32, 32, 1]` | `[B, 32, 32, 1, 1]` | modes=8, width=16 |
| 1D Burgers | UNet1d | `1D_Burgers_Sols_Nu1.0_Unet_small.pt` | `[B, 5, 64]` | `[B, 1, 64]` | initial_step=5, reduced_resolution=16 |
| 1D Burgers | FNO1d | `1D_Burgers_Sols_Nu1.0_FNO_small.pt` | `[B, 64, 5]` | `[B, 64, 1, 1]` | modes=8, width=16, initial_step=5 |
| 1D Diff-Sorp | UNet1d | `1D_diff-sorp_NA_NA_Unet_small.pt` | `[B, 5, 64]` | `[B, 1, 64]` | initial_step=5, reduced_resolution=16 |
| 1D Diff-Sorp | FNO1d | `1D_diff-sorp_NA_NA_FNO_small.pt` | `[B, 64, 5]` | `[B, 64, 1, 1]` | modes=8, width=16, initial_step=5 |
| 1D Diff-Sorp | **PINN1d** | `1D_diff-sorp_NA_NA_0001.h5_PINN.pt-15000.pt` | `[N, 2]` | `[N, 1]` | hidden=40, layers=7 |

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
predictor = PDEPredictor.load("2D_DarcyFlow_beta1.0_Train_Unet_small.pt")

# 查看期望输入格式
print(predictor)
# PDEPredictor(
#   model_type=UNet2d,
#   input_shape=[B, 1, Nx, Ny],
#   out_channels=1,
#   device=cpu
# )

# 构造输入数据
permeability = np.random.uniform(0.1, 1.0, (1, 1, 32, 32)).astype(np.float32)

# 推理
pressure = predictor.predict(permeability)
print(f"Output shape: {pressure.shape}")  # (1, 1, 32, 32)
```

---

## 6. 输入尺寸限制（重要）

模型对输入尺寸有严格限制，不符合要求的输入会导致运行时错误。

### 6.1 各维度限制

| 维度 | 是否可变 | 限制说明 |
|------|----------|----------|
| **Batch (B)** | ✓ 可变 | 任意正整数 |
| **通道 (C / T₀)** | ✗ 固定 | 必须与模型训练时一致 |
| **空间 (Nx, Ny)** | ✓ 可变 | 需满足下述约束 |

### 6.2 空间尺寸约束

| 模型类型 | 约束条件 | 原因 |
|----------|----------|------|
| **UNet** | Nx, Ny 必须能被 **16** 整除 | 4层池化操作，每层尺寸减半 |
| **FNO** | Nx, Ny 必须 ≥ **2 × modes** | FFT 频域尺寸需 ≥ modes 参数 |
| **PINN** | **无限制** | 点查询方式，N 可为任意正整数 |

### 6.3 各模型具体要求

| 模型 | 通道数 | 空间尺寸要求 | 有效尺寸示例 |
|------|--------|-------------|-------------|
| UNet2d (Darcy) | C=1 | Nx, Ny ∈ {16, 32, 48, 64, ...} | 32, 64, 128 |
| FNO2d (Darcy, modes=8) | C=1 | Nx, Ny ≥ 16 | 16, 32, 64 |
| UNet1d (Burgers) | T₀=5 | Nx ∈ {16, 32, 48, 64, ...} | 32, 64, 128 |
| FNO1d (Burgers, modes=8) | T₀=5 | Nx ≥ 16 | 16, 32, 64 |
| UNet1d (Diff-Sorp) | T₀=5 | Nx ∈ {16, 32, 48, 64, ...} | 32, 64, 128 |
| FNO1d (Diff-Sorp, modes=8) | T₀=5 | Nx ≥ 16 | 16, 32, 64 |
| PINN1d (Diff-Sorp) | 2 (x,t) | **无限制** | 任意 N (如 100, 1024, 10000) |

### 6.4 示例：有效与无效输入

```python
from pdebench_inference import PDEPredictor
import numpy as np

predictor_unet = PDEPredictor.load("2D_DarcyFlow_beta1.0_Train_Unet_small.pt")
predictor_fno = PDEPredictor.load("2D_DarcyFlow_beta1.0_Train_FNO_small.pt")
predictor_pinn = PDEPredictor.load("1D_diff-sorp_NA_NA_0001.h5_PINN.pt-15000.pt")

# ✓ 有效输入
predictor_unet.predict(np.zeros((1, 1, 32, 32)))   # 32 能被 16 整除
predictor_unet.predict(np.zeros((1, 1, 64, 64)))   # 64 能被 16 整除
predictor_fno.predict(np.zeros((1, 32, 32, 1)))    # 32 >= 16
predictor_fno.predict(np.zeros((1, 64, 64, 1)))    # 64 >= 16，FNO 支持任意尺寸

# PINN: 任意数量的查询点都有效
predictor_pinn.predict(np.zeros((100, 2)))         # 100 个点
predictor_pinn.predict(np.zeros((10000, 2)))       # 10000 个点
predictor_pinn.predict(np.zeros((1, 2)))           # 单个点也可以

# ✗ 无效输入
predictor_unet.predict(np.zeros((1, 1, 100, 100))) # 100 不能被 16 整除
predictor_unet.predict(np.zeros((1, 2, 32, 32)))   # 通道数 2 != 1
predictor_fno.predict(np.zeros((1, 10, 10, 1)))    # 10 < 16
predictor_pinn.predict(np.zeros((100, 3)))         # PINN 期望 2 维输入 (x, t)
```

---

## 7. 数据预处理

### 7.1 空间分辨率下采样

预训练模型使用了下采样的数据进行训练：

| 任务 | 原始分辨率 | 训练分辨率 | 下采样因子 |
|------|-----------|-----------|-----------|
| 2D Darcy Flow | 128×128 | 32×32 | 4 |
| 1D Burgers | 1024 | 64 | 16 |
| 1D Diff-Sorp | 1024 | 64 | 16 |

```python
# 下采样示例
def downsample_1d(data, factor):
    return data[..., ::factor]

def downsample_2d(data, factor):
    return data[..., ::factor, ::factor]
```

### 7.2 时间分辨率下采样（用于时序任务）

| 任务 | 原始时间步 | 训练时间步 | 下采样因子 | 输入时间步 |
|------|-----------|-----------|-----------|-----------|
| 1D Burgers | 201 | 41 | 5 | 5 |
| 1D Diff-Sorp | 101 | 51 | 2 | 5 |

---

## 8. API 参考

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

### `run_pinn_inference(predictor, dataset, sample_indices=None, save_predictions=True, show_progress=True)`

PINN 模型专用推理函数。

**参数**：
- `predictor`: `PDEPredictor` 实例（必须是 PINN1d 模型）
- `dataset`: `DiffSorpDataset` 实例
- `sample_indices`: 可选，要评估的样本索引列表
- `save_predictions`: 是否保存预测结果
- `show_progress`: 是否显示进度条

**返回**：`InferenceResult` 实例，包含 MSE、MAE 等指标

**示例**：
```python
from pdebench_inference import PDEPredictor, DiffSorpDataset, run_pinn_inference

predictor = PDEPredictor.load("1D_diff-sorp_NA_NA_0001.h5_PINN.pt-15000.pt")
dataset = DiffSorpDataset("1D_diff-sorp_NA_NA.h5")

result = run_pinn_inference(predictor, dataset, sample_indices=list(range(100)))
print(f"MSE: {result.mse:.6e}")
```

---

## 9. 依赖要求

```
torch>=1.9.0
numpy>=1.19.0
```

脚本为自包含设计，无需安装 PDEBench 或其他额外依赖。
