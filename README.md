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

跑通专家模型的推理，有两种方式可以选择：

1. **如果要测试模型在整个数据集上的表现，则需要使用 `PDEBench` 项目的代码（该目录中不提供）：**
   
   1. Clone PDEBench仓库到本地 
      ```bash
      git clone https://github.com/pdebench/PDEBench.git
      ```
   2. 按照PDEBench中的 `README.md` 的指导安装依赖；
   3. 由于PDEBench项目中U-Net推理代码有问题，所以你需要用当前目录下 `PDEBench_patch/train.py` `PDEBench_patch/utils.py` 替换PDEBench项目目录（`PDEBench/`）中的 `pdebench/models/unet/train.py` 和 `pdebench/models/unet/utils.py`；
   4. 将当前目录下 `PDEBench_patch/run_all_eval.sh` `PDEBench_patch/eval_darcy_unet.py` `PDEBench_patch/eval_pinn_diff_sorp.py` 复制到PDEBench项目目录（`PDEBench/`）下；
   5. 将上面提到的所有数据集与模型放在PDEBench项目目录（`PDEBench/`）下，不要修改文件名；
   6. 运行 `run_all_eval.sh` ，它会用完整的数据集来评估所有模型：
      ```bash
      chmod +x run_all_eval.sh
      ./run_all_eval.sh
      ```
   7. 运行结束后会生成每一个模型的评估结果（`*.pickle`）。
   
   **测试环境**：NVIDIA RTX 4090 (24GB), Python 3.9, Ubuntu 22.04, CUDA 12.8, PyTorch 2.8.0
   **运行时间**：约 3 min

<br>

2. **在PiERN中，我们不需要专家模型进行批量的推理，也不需要读取数据集。因此，为了与其他模块整合起来，我们写了一个轻量化、可以被模块化调用的推理API，即 `pdebench_inference.py`。这个文件不依赖于PDEBench项目的代码。使用说明见 [INFERENCE_API.md](INFERENCE_API.md)**

## 3. 评估结果

专家模型运行结果（包含 `*.pickle` 与 `Results.pdf`）见 `eval_results/`。