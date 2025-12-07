from __future__ import annotations

"""
Evaluate the pretrained U-Net for the 2D DarcyFlow dataset.

This script:
 1) Loads the 2D DarcyFlow dataset via the existing UNetDatasetSingle.
 2) Reconstructs a UNet2d model whose input/output channels are inferred
    from the checkpoint file.
 3) Runs autoregressive rollouts and computes the standard PDEBench metrics.

Usage (from the PDEBench repo root):

    python eval_darcy_unet.py

Defaults are set to match the files that are present in this project:
  - data:  2D_DarcyFlow_beta1.0_Train.hdf5
  - ckpt:  2D_DarcyFlow_beta1.0_Train_Unet_PF_1.pt
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from pdebench.models.metrics import metrics
from pdebench.models.unet.unet import UNet2d
from pdebench.models.unet.utils import UNetDatasetSingle


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def infer_channels(ckpt_path: Path) -> tuple[int, int]:
    """Infer in/out channels of the UNet2d from a checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"]

    # First encoder conv determines input channels
    enc1_w = state_dict["encoder1.enc1conv1.weight"]
    in_channels = enc1_w.shape[1]

    # Final conv determines output channels
    out_w = state_dict["conv.weight"]
    out_channels = out_w.shape[0]

    return int(in_channels), int(out_channels)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the 2D DarcyFlow U-Net model."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="2D_DarcyFlow_beta1.0_Train.hdf5",
        help="Path to DarcyFlow HDF5 file (relative to this file).",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="2D_DarcyFlow_beta1.0_Train_Unet_PF_1.pt",
        help="Path to the pretrained U-Net checkpoint.",
    )
    parser.add_argument(
        "--initial_step",
        type=int,
        default=1,
        help="Number of initial time steps used as context.",
    )
    parser.add_argument(
        "--reduced_resolution",
        type=int,
        default=2,
        help="Spatial resolution reduction factor (must match training config).",
    )
    parser.add_argument(
        "--reduced_resolution_t",
        type=int,
        default=1,
        help="Temporal resolution reduction factor (must match training config).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    data_path = (root / args.data).resolve()
    ckpt_path = (root / args.ckpt).resolve()

    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Using dataset:   {data_path}")
    print(f"Using checkpoint: {ckpt_path}")

    # Dataset and loader (single HDF5 file, test split)
    # NOTE: reduced_resolution must match the value used during training!
    # From config_Darcy.yaml: reduced_resolution=2
    ds = UNetDatasetSingle(
        filename=data_path.name,
        saved_folder=str(data_path.parent) + '/',
        initial_step=args.initial_step,
        reduced_resolution=args.reduced_resolution,
        reduced_resolution_t=args.reduced_resolution_t,
        reduced_batch=1,
        if_test=True,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    # Inspect one sample to determine spatial/temporal dims
    _, sample = next(iter(loader))
    dims = len(sample.shape)
    if dims != 5:
        msg = f"Expected 2D data with shape [B, Nx, Ny, T, C], got {sample.shape}"
        raise RuntimeError(msg)

    # Infer channel counts from checkpoint and build model accordingly
    in_channels, out_channels = infer_channels(ckpt_path)
    model = UNet2d(in_channels * args.initial_step, out_channels).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Run metrics (autoregressive rollout as in metrics(mode="Unet"))
    Lx = 1.0
    Ly = 1.0
    Lz = 1.0
    errs = metrics(
        val_loader=loader,
        model=model,
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
        plot=False,
        channel_plot=0,
        model_name=ckpt_path.stem,
        x_min=-1.0,
        x_max=1.0,
        y_min=-1.0,
        y_max=1.0,
        t_min=0.0,
        t_max=5.0,
        mode="Unet",
        initial_step=args.initial_step,
    )

    errors_np = []
    for e in errs:
        if hasattr(e, "detach"):
            e_np = e.detach().cpu().numpy()
        else:
            e_np = np.array(e)
        errors_np.append(e_np)

    out_path = ckpt_path.with_name(ckpt_path.stem + "_eval.pickle")
    with out_path.open("wb") as f:
        pickle.dump(errors_np, f)

    print("Evaluation finished.")
    print(f"Metrics saved to: {out_path}")


if __name__ == "__main__":
    main()

