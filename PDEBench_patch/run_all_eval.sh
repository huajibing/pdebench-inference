#!/usr/bin/env bash
set -e

# Run evaluation (inference + metrics) for the models that are present
# in this project directory, using the official PDEBench code paths.
#
# This script assumes that:
#   - You are running it from the PDEBench repo root, OR
#   - You call it via:  (cd PDEBench && bash run_all_eval.sh)
#   - PDEBench (and its dependencies) are already installed:
#       pip install .

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

echo "Running evaluations from: ${ROOT_DIR}"

###############################################################################
# 1D Burgers: FNO + U-Net (PF-20)
# Official model parameters:
#   - FNO: initial_step=10, width=20, modes=12
#   - U-Net PF-20: initial_step=10, in_channels=10, out_channels=1, ar_mode=True, pushforward=True
###############################################################################

if [ -f "1D_Burgers_Sols_Nu1.0.hdf5" ]; then
  if [ -f "1D_Burgers_Sols_Nu1.0_FNO.pt" ]; then
    echo "==> 1D Burgers (FNO)"
    python -m pdebench.models.train_models_inverse \
      +args=config_Bgs.yaml \
      ++args.model_name=FNO \
      ++args.if_training=False \
      ++args.filename='1D_Burgers_Sols_Nu1.0.hdf5' \
      ++args.single_file=True \
      ++args.base_path='./' \
      ++args.initial_step=10 \
      ++args.reduced_resolution=4 \
      ++args.reduced_resolution_t=5
  else
    echo "!! Skipping 1D Burgers FNO: checkpoint file not found."
  fi

  if [ -f "1D_Burgers_Sols_Nu1.0_Unet-PF-20.pt" ]; then
    echo "==> 1D Burgers (U-Net, PF-20)"
    python -m pdebench.models.train_models_inverse \
      +args=config_Bgs.yaml \
      ++args.model_name=Unet \
      ++args.if_training=False \
      ++args.filename='1D_Burgers_Sols_Nu1.0.hdf5' \
      ++args.single_file=True \
      ++args.base_path='./' \
      ++args.initial_step=10 \
      ++args.in_channels=1 \
      ++args.out_channels=1 \
      ++args.reduced_resolution=4 \
      ++args.reduced_resolution_t=5 \
      ++args.ar_mode=True \
      ++args.pushforward=True \
      ++args.unroll_step=20
  else
    echo "!! Skipping 1D Burgers U-Net: checkpoint file not found."
  fi
else
  echo "!! Skipping 1D Burgers: data file 1D_Burgers_Sols_Nu1.0.hdf5 not found."
fi

###############################################################################
# 1D diffusion-sorption: FNO + U-Net (1-step) + U-Net (PF-20)
# Official model parameters:
#   - FNO: initial_step=10, width=64, modes=16
#   - U-Net 1-step: initial_step=10, ar_mode=False, pushforward=False
#   - U-Net PF-20: initial_step=10, ar_mode=True, pushforward=True
###############################################################################

if [ -f "1D_diff-sorp_NA_NA.h5" ]; then
  if [ -f "1D_diff-sorp_NA_NA_FNO.pt" ]; then
    echo "==> 1D diffusion-sorption (FNO)"
    python -m pdebench.models.train_models_forward \
      +args=config_diff-sorp.yaml \
      ++args.model_name=FNO \
      ++args.if_training=False \
      ++args.filename='1D_diff-sorp_NA_NA' \
      ++args.single_file=False \
      ++args.data_path='./' \
      ++args.initial_step=10
  else
    echo "!! Skipping 1D diffusion-sorption FNO: checkpoint file not found."
  fi

  if [ -f "1D_diff-sorp_NA_NA_Unet-1-step.pt" ]; then
    echo "==> 1D diffusion-sorption (U-Net, 1-step)"
    python -m pdebench.models.train_models_forward \
      +args=config_diff-sorp.yaml \
      ++args.model_name=Unet \
      ++args.if_training=False \
      ++args.filename='1D_diff-sorp_NA_NA' \
      ++args.single_file=False \
      ++args.data_path='./' \
      ++args.initial_step=10 \
      ++args.ar_mode=False \
      ++args.pushforward=False
  else
    echo "!! Skipping 1D diffusion-sorption U-Net 1-step: checkpoint file not found."
  fi

  if [ -f "1D_diff-sorp_NA_NA_Unet-PF-20.pt" ]; then
    echo "==> 1D diffusion-sorption (U-Net, PF-20)"
    python -m pdebench.models.train_models_forward \
      +args=config_diff-sorp.yaml \
      ++args.model_name=Unet \
      ++args.if_training=False \
      ++args.filename='1D_diff-sorp_NA_NA' \
      ++args.single_file=False \
      ++args.data_path='./' \
      ++args.initial_step=10 \
      ++args.in_channels=1 \
      ++args.out_channels=1 \
      ++args.ar_mode=True \
      ++args.pushforward=True \
      ++args.unroll_step=20
  else
    echo "!! Skipping 1D diffusion-sorption U-Net PF-20: checkpoint file not found."
  fi
else
  echo "!! Skipping 1D diffusion-sorption: data file 1D_diff-sorp_NA_NA.h5 not found."
fi

###############################################################################
# 2D DarcyFlow: FNO + U-Net
# Official model parameters:
#   - FNO: initial_step=1 (input_dim=3 = 1*1+2 for 2D grid)
#   - U-Net: initial_step=1, in_channels=1, out_channels=1
###############################################################################

if [ -f "2D_DarcyFlow_beta1.0_Train.hdf5" ]; then
  if [ -f "2D_DarcyFlow_beta1.0_Train_FNO.pt" ]; then
    echo "==> 2D DarcyFlow (FNO)"
    python -m pdebench.models.train_models_forward \
      +args=config_Darcy.yaml \
      ++args.model_name=FNO \
      ++args.if_training=False \
      ++args.filename='2D_DarcyFlow_beta1.0_Train.hdf5' \
      ++args.single_file=True \
      ++args.data_path='./' \
      ++args.initial_step=1
  else
    echo "!! Skipping 2D DarcyFlow FNO: checkpoint file not found."
  fi

  if [ -f "2D_DarcyFlow_beta1.0_Train_Unet_PF_1.pt" ]; then
    echo "==> 2D DarcyFlow (U-Net, custom eval)"
    # NOTE: reduced_resolution=2 and initial_step=1 must match training config
    python eval_darcy_unet.py \
      --data 2D_DarcyFlow_beta1.0_Train.hdf5 \
      --ckpt 2D_DarcyFlow_beta1.0_Train_Unet_PF_1.pt \
      --initial_step 1 \
      --reduced_resolution 2 \
      --reduced_resolution_t 1
  else
    echo "!! Skipping 2D DarcyFlow U-Net: checkpoint file not found."
  fi
else
  echo "!! Skipping 2D DarcyFlow: data file 2D_DarcyFlow_beta1.0_Train.hdf5 not found."
fi

###############################################################################
# PINN (1D diffusion-sorption) - optional
###############################################################################

if [ -f "1D_diff-sorp_NA_NA.h5" ] && \
   [ -f "1D_diff-sorp_NA_NA_0001.h5_PINN.pt-15000.pt" ]; then
  echo "==> 1D diffusion-sorption (PINN)"
  python eval_pinn_diff_sorp.py \
    --data 1D_diff-sorp_NA_NA.h5 \
    --ckpt 1D_diff-sorp_NA_NA_0001.h5_PINN.pt-15000.pt \
    --seed 0001
else
  echo "!! Skipping 1D diffusion-sorption PINN: data or checkpoint not found."
fi

echo "All requested evaluations finished."
