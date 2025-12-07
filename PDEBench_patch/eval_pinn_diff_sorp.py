from __future__ import annotations

"""
Evaluate the pretrained PINN for the 1D diffusion–sorption equation.

This script:
 1) Reconstructs the PINN model architecture used in PDEBench.
 2) Loads a checkpoint (no further training).
 3) Uses a single seed from the diffusion–sorption dataset to build
    test inputs and ground truth.
 4) Computes the standard PDEBench error metrics and saves them.

Usage (from the PDEBench repo root):

    python eval_pinn_diff_sorp.py \
        --data 1D_diff-sorp_NA_NA.h5 \
        --ckpt 1D_diff-sorp_NA_NA_0001.h5_PINN.pt-15000.pt \
        --seed 0001

Default arguments are chosen to match the files that are present in this
project directory, so in most cases you can simply run:

    python eval_pinn_diff_sorp.py
"""

import argparse
import pickle
from pathlib import Path

import deepxde as dde
import h5py
import numpy as np
import torch

from pdebench.models.metrics import metric_func
from pdebench.models.pinn.pde_definitions import pde_diffusion_sorption


def build_model() -> dde.Model:
    """Rebuild the diffusion–sorption PINN model (geometry + PDE + network)."""
    # Geometry and time domain (must match training setup)
    geom = dde.geometry.Interval(0.0, 1.0)
    timedomain = dde.geometry.TimeDomain(0.0, 500.0)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    # Boundary and initial conditions (same as in setup_diffusion_sorption)
    D = 5e-4

    ic = dde.icbc.IC(
        geomtime,
        lambda x: 0.0,
        lambda _, on_boundary: on_boundary,
    )

    bc_left = dde.icbc.DirichletBC(
        geomtime,
        lambda x: 1.0,
        lambda x, on_boundary: on_boundary and np.isclose(x[0], 0.0),
    )

    def operator_bc(inputs, outputs, _X):
        du_x = dde.grad.jacobian(outputs, inputs, i=0, j=0)
        return outputs - D * du_x

    bc_right = dde.icbc.OperatorBC(
        geomtime,
        operator_bc,
        lambda x, on_boundary: on_boundary and np.isclose(x[0], 1.0),
    )

    data = dde.data.TimePDE(
        geomtime,
        pde_diffusion_sorption,
        [ic, bc_left, bc_right],
        num_domain=1000,
        num_boundary=1000,
        num_initial=5000,
    )

    # Network architecture used in PDEBench for this problem
    net = dde.nn.FNN([2] + [40] * 6 + [1], "tanh", "Glorot normal")

    def transform_output(_x, y):
        # Enforce non-negativity of the solution
        return torch.relu(y)

    net.apply_output_transform(transform_output)

    model = dde.Model(data, net)
    return model


def load_test_data(
    h5_path: Path, seed: str, n_last_time_steps: int = 20
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    Build test inputs and outputs from the diffusion–sorption dataset.

    Returns:
        test_input:  (N, 2)  torch tensor with columns [x, t]
        test_output: (Nx, T_last, 1) torch tensor with u(x, t)
        Nx:          spatial resolution
        T_last:      number of time steps used (<= n_last_time_steps)
    """
    with h5py.File(h5_path, "r") as f:
        if seed not in f:
            raise KeyError(f"Seed '{seed}' not found in {h5_path}")
        g = f[seed]
        x = np.array(g["grid"]["x"], dtype=np.float32)  # (Nx,)
        t = np.array(g["grid"]["t"], dtype=np.float32)  # (Nt,)
        data = np.array(g["data"], dtype=np.float32)  # (Nt, Nx, 1)

    nx = x.shape[0]
    nt = t.shape[0]
    t_last = min(n_last_time_steps, nt)

    # Meshgrid for the full (x, t) domain
    XX, TT = np.meshgrid(x, t, indexing="ij")  # (Nx, Nt)

    # Select only the last t_last time steps for evaluation
    XX_last = XX[:, -t_last:]
    TT_last = TT[:, -t_last:]

    test_input = np.vstack([XX_last.ravel(), TT_last.ravel()]).T  # (Nx * t_last, 2)

    # data is [Nt, Nx, 1]; convert to [Nx, Nt, 1] and slice last t_last steps
    data_xt = np.transpose(data, (1, 0, 2))  # (Nx, Nt, 1)
    data_last = data_xt[:, -t_last:, :]  # (Nx, t_last, 1)

    test_input_t = torch.from_numpy(test_input.astype(np.float32))
    test_output_t = torch.from_numpy(data_last.astype(np.float32))
    return test_input_t, test_output_t, nx, t_last


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the PINN for 1D diffusion–sorption."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="1D_diff-sorp_NA_NA.h5",
        help="Path to diffusion–sorption HDF5 dataset (relative to this file).",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="1D_diff-sorp_NA_NA_0001.h5_PINN.pt-15000.pt",
        help="Path to the pretrained PINN checkpoint.",
    )
    parser.add_argument(
        "--seed",
        type=str,
        default="0001",
        help="Seed ID inside the HDF5 file to use as ground truth.",
    )
    parser.add_argument(
        "--n_last",
        type=int,
        default=20,
        help="Number of last time steps to evaluate on.",
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
    print(f"Using seed:      {args.seed}")

    # Build and restore model
    model = build_model()
    model.compile("adam", lr=1e-3)
    model.restore(str(ckpt_path))

    # Build test inputs/targets from data
    test_input, test_output, nx, t_last = load_test_data(
        data_path, args.seed, n_last_time_steps=args.n_last
    )

    # DeepXDE expects numpy arrays for prediction
    pred_np = model.predict(test_input.numpy())
    # Only 1 component is present for this problem
    pred = torch.from_numpy(pred_np[:, :1])

    # Reshape to [batch=1, Nx, T, C] for metric_func
    pred = pred.reshape(1, nx, t_last, 1)
    target = test_output.reshape(1, nx, t_last, 1)

    errs = metric_func(pred, target)
    errors_np = [e.detach().cpu().numpy() for e in errs]

    # Save metrics next to the checkpoint
    out_name = ckpt_path.stem + "_eval.pickle"
    out_path = ckpt_path.with_name(out_name)
    with out_path.open("wb") as f:
        pickle.dump(errors_np, f)

    print("Evaluation finished.")
    print(f"Metrics saved to: {out_path}")


if __name__ == "__main__":
    main()

