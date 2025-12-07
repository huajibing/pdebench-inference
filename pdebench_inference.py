"""
PDEBench Unified Inference Script

A lightweight, self-contained inference script for PDEBench pretrained models.
Supports FNO and U-Net models for 1D Burgers, 1D diffusion-sorption, and 2D Darcy Flow.

This script intentionally embeds all necessary model definitions to minimize
dependencies on the PDEBench codebase.

Usage:
    from pdebench_inference import PDEPredictor

    # Load a pretrained model
    predictor = PDEPredictor.load("2D_DarcyFlow_beta1.0_Train_Unet_PF_1.pt")

    # Run inference on input data
    output = predictor.predict(input_data)

    # Or use the convenience function
    output = predict_pde(
        checkpoint_path="2D_DarcyFlow_beta1.0_Train_Unet_PF_1.pt",
        input_data=input_data
    )
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


# =============================================================================
# Model Definitions (self-contained, no external dependencies)
# =============================================================================

class SpectralConv1d(nn.Module):
    """1D Fourier layer: FFT -> linear transform -> Inverse FFT."""

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes  # Use modes1 to match checkpoint naming
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-1) // 2 + 1,
            device=x.device, dtype=torch.cfloat
        )
        out_ft[:, :, :self.modes1] = torch.einsum(
            "bix,iox->box", x_ft[:, :, :self.modes1], self.weights1
        )
        return torch.fft.irfft(out_ft, n=x.size(-1))


class SpectralConv2d(nn.Module):
    """2D Fourier layer: FFT -> linear transform -> Inverse FFT."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class FNO1d(nn.Module):
    """1D Fourier Neural Operator."""

    def __init__(
        self,
        num_channels: int = 1,
        modes: int = 16,
        width: int = 64,
        initial_step: int = 10,
    ):
        super().__init__()
        self.modes = modes
        self.width = width
        self.initial_step = initial_step
        self.num_channels = num_channels
        self.padding = 2

        self.fc0 = nn.Linear(initial_step * num_channels + 1, width)
        self.conv0 = SpectralConv1d(width, width, modes)
        self.conv1 = SpectralConv1d(width, width, modes)
        self.conv2 = SpectralConv1d(width, width, modes)
        self.conv3 = SpectralConv1d(width, width, modes)
        self.w0 = nn.Conv1d(width, width, 1)
        self.w1 = nn.Conv1d(width, width, 1)
        self.w2 = nn.Conv1d(width, width, 1)
        self.w3 = nn.Conv1d(width, width, 1)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, num_channels)

    def forward(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        # x: [B, Nx, initial_step * num_channels], grid: [B, Nx, 1]
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        x = F.pad(x, [0, self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding]
        x = x.permute(0, 2, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x.unsqueeze(-2)  # [B, Nx, 1, num_channels]


class FNO2d(nn.Module):
    """2D Fourier Neural Operator."""

    def __init__(
        self,
        num_channels: int = 1,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 20,
        initial_step: int = 10,
    ):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.initial_step = initial_step
        self.num_channels = num_channels
        self.padding = 2

        self.fc0 = nn.Linear(initial_step * num_channels + 2, width)
        self.conv0 = SpectralConv2d(width, width, modes1, modes2)
        self.conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.conv2 = SpectralConv2d(width, width, modes1, modes2)
        self.conv3 = SpectralConv2d(width, width, modes1, modes2)
        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, num_channels)

    def forward(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        # x: [B, Nx, Ny, initial_step * num_channels], grid: [B, Nx, Ny, 2]
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0, self.padding, 0, self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x.unsqueeze(-2)  # [B, Nx, Ny, 1, num_channels]


def _unet_block_1d(in_channels: int, features: int, name: str) -> nn.Sequential:
    return nn.Sequential(OrderedDict([
        (f"{name}conv1", nn.Conv1d(in_channels, features, kernel_size=3, padding=1, bias=False)),
        (f"{name}norm1", nn.BatchNorm1d(features)),
        (f"{name}tanh1", nn.Tanh()),
        (f"{name}conv2", nn.Conv1d(features, features, kernel_size=3, padding=1, bias=False)),
        (f"{name}norm2", nn.BatchNorm1d(features)),
        (f"{name}tanh2", nn.Tanh()),
    ]))


def _unet_block_2d(in_channels: int, features: int, name: str) -> nn.Sequential:
    return nn.Sequential(OrderedDict([
        (f"{name}conv1", nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False)),
        (f"{name}norm1", nn.BatchNorm2d(features)),
        (f"{name}tanh1", nn.Tanh()),
        (f"{name}conv2", nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False)),
        (f"{name}norm2", nn.BatchNorm2d(features)),
        (f"{name}tanh2", nn.Tanh()),
    ]))


class UNet1d(nn.Module):
    """1D U-Net for time-dependent PDEs."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1, init_features: int = 32):
        super().__init__()
        f = init_features
        self.encoder1 = _unet_block_1d(in_channels, f, "enc1")
        self.pool1 = nn.MaxPool1d(2, 2)
        self.encoder2 = _unet_block_1d(f, f * 2, "enc2")
        self.pool2 = nn.MaxPool1d(2, 2)
        self.encoder3 = _unet_block_1d(f * 2, f * 4, "enc3")
        self.pool3 = nn.MaxPool1d(2, 2)
        self.encoder4 = _unet_block_1d(f * 4, f * 8, "enc4")
        self.pool4 = nn.MaxPool1d(2, 2)
        self.bottleneck = _unet_block_1d(f * 8, f * 16, "bottleneck")
        self.upconv4 = nn.ConvTranspose1d(f * 16, f * 8, 2, 2)
        self.decoder4 = _unet_block_1d(f * 16, f * 8, "dec4")
        self.upconv3 = nn.ConvTranspose1d(f * 8, f * 4, 2, 2)
        self.decoder3 = _unet_block_1d(f * 8, f * 4, "dec3")
        self.upconv2 = nn.ConvTranspose1d(f * 4, f * 2, 2, 2)
        self.decoder2 = _unet_block_1d(f * 4, f * 2, "dec2")
        self.upconv1 = nn.ConvTranspose1d(f * 2, f, 2, 2)
        self.decoder1 = _unet_block_1d(f * 2, f, "dec1")
        self.conv = nn.Conv1d(f, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, Nx]
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(enc4))
        dec4 = torch.cat([self.upconv4(bottleneck), enc4], dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = torch.cat([self.upconv3(dec4), enc3], dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = torch.cat([self.upconv2(dec3), enc2], dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = torch.cat([self.upconv1(dec2), enc1], dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1)  # [B, out_channels, Nx]


class UNet2d(nn.Module):
    """2D U-Net for spatial PDEs."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1, init_features: int = 32):
        super().__init__()
        f = init_features
        self.encoder1 = _unet_block_2d(in_channels, f, "enc1")
        self.pool1 = nn.MaxPool2d(2, 2)
        self.encoder2 = _unet_block_2d(f, f * 2, "enc2")
        self.pool2 = nn.MaxPool2d(2, 2)
        self.encoder3 = _unet_block_2d(f * 2, f * 4, "enc3")
        self.pool3 = nn.MaxPool2d(2, 2)
        self.encoder4 = _unet_block_2d(f * 4, f * 8, "enc4")
        self.pool4 = nn.MaxPool2d(2, 2)
        self.bottleneck = _unet_block_2d(f * 8, f * 16, "bottleneck")
        self.upconv4 = nn.ConvTranspose2d(f * 16, f * 8, 2, 2)
        self.decoder4 = _unet_block_2d(f * 16, f * 8, "dec4")
        self.upconv3 = nn.ConvTranspose2d(f * 8, f * 4, 2, 2)
        self.decoder3 = _unet_block_2d(f * 8, f * 4, "dec3")
        self.upconv2 = nn.ConvTranspose2d(f * 4, f * 2, 2, 2)
        self.decoder2 = _unet_block_2d(f * 4, f * 2, "dec2")
        self.upconv1 = nn.ConvTranspose2d(f * 2, f, 2, 2)
        self.decoder1 = _unet_block_2d(f * 2, f, "dec1")
        self.conv = nn.Conv2d(f, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, Nx, Ny]
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(enc4))
        dec4 = torch.cat([self.upconv4(bottleneck), enc4], dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = torch.cat([self.upconv3(dec4), enc3], dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = torch.cat([self.upconv2(dec3), enc2], dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = torch.cat([self.upconv1(dec2), enc1], dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1)  # [B, out_channels, Nx, Ny]


# =============================================================================
# Model Configuration and Inference
# =============================================================================

ModelType = Literal["FNO1d", "FNO2d", "UNet1d", "UNet2d"]


@dataclass
class ModelConfig:
    """Model configuration inferred from checkpoint."""
    model_type: ModelType
    in_channels: int
    out_channels: int
    # FNO specific
    width: int | None = None
    modes: int | None = None
    modes1: int | None = None
    modes2: int | None = None
    initial_step: int | None = None
    num_channels: int | None = None


def infer_model_config(state_dict: dict[str, torch.Tensor]) -> ModelConfig:
    """Infer model architecture and parameters from checkpoint state dict."""
    keys = set(state_dict.keys())

    # Detect model type
    is_fno = "fc0.weight" in keys and "conv0.weights1" in keys
    is_unet = "encoder1.enc1conv1.weight" in keys

    if is_fno:
        # FNO model
        fc0_w = state_dict["fc0.weight"]  # [width, input_dim]
        fc2_w = state_dict["fc2.weight"]  # [num_channels, 128]
        conv0_w1 = state_dict["conv0.weights1"]

        width = fc0_w.shape[0]
        input_dim = fc0_w.shape[1]
        num_channels = fc2_w.shape[0]

        if conv0_w1.dim() == 3:
            # FNO1d: [width, width, modes]
            modes = conv0_w1.shape[2]
            initial_step = (input_dim - 1) // num_channels  # -1 for grid
            return ModelConfig(
                model_type="FNO1d",
                in_channels=initial_step * num_channels,
                out_channels=num_channels,
                width=width,
                modes=modes,
                initial_step=initial_step,
                num_channels=num_channels,
            )
        else:
            # FNO2d: [width, width, modes1, modes2]
            modes1 = conv0_w1.shape[2]
            modes2 = conv0_w1.shape[3]
            initial_step = (input_dim - 2) // num_channels  # -2 for 2D grid
            return ModelConfig(
                model_type="FNO2d",
                in_channels=initial_step * num_channels,
                out_channels=num_channels,
                width=width,
                modes1=modes1,
                modes2=modes2,
                initial_step=initial_step,
                num_channels=num_channels,
            )

    elif is_unet:
        # UNet model
        enc1_w = state_dict["encoder1.enc1conv1.weight"]
        conv_w = state_dict["conv.weight"]

        if enc1_w.dim() == 3:
            # UNet1d: [features, in_channels, kernel_size]
            in_channels = enc1_w.shape[1]
            out_channels = conv_w.shape[0]
            return ModelConfig(
                model_type="UNet1d",
                in_channels=in_channels,
                out_channels=out_channels,
            )
        else:
            # UNet2d: [features, in_channels, k, k]
            in_channels = enc1_w.shape[1]
            out_channels = conv_w.shape[0]
            return ModelConfig(
                model_type="UNet2d",
                in_channels=in_channels,
                out_channels=out_channels,
            )

    raise ValueError("Unknown model architecture in checkpoint")


def build_model(config: ModelConfig) -> nn.Module:
    """Build a model instance from configuration."""
    if config.model_type == "FNO1d":
        return FNO1d(
            num_channels=config.num_channels,
            modes=config.modes,
            width=config.width,
            initial_step=config.initial_step,
        )
    elif config.model_type == "FNO2d":
        return FNO2d(
            num_channels=config.num_channels,
            modes1=config.modes1,
            modes2=config.modes2,
            width=config.width,
            initial_step=config.initial_step,
        )
    elif config.model_type == "UNet1d":
        return UNet1d(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
        )
    elif config.model_type == "UNet2d":
        return UNet2d(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
        )
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")


# =============================================================================
# PDEPredictor: Main Interface
# =============================================================================

class PDEPredictor:
    """
    Unified predictor for PDEBench pretrained models.

    Example:
        predictor = PDEPredictor.load("2D_DarcyFlow_beta1.0_Train_Unet_PF_1.pt")
        output = predictor.predict(input_data)
    """

    def __init__(
        self,
        model: nn.Module,
        config: ModelConfig,
        device: torch.device | str = "cpu",
    ):
        self.model = model
        self.config = config
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def load(
        cls,
        checkpoint_path: str | Path,
        device: torch.device | str | None = None,
    ) -> "PDEPredictor":
        """
        Load a pretrained model from checkpoint.

        Args:
            checkpoint_path: Path to the .pt checkpoint file
            device: Device to load the model on (default: cuda if available, else cpu)

        Returns:
            PDEPredictor instance ready for inference
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model_state_dict"]

        # Infer model configuration and build model
        config = infer_model_config(state_dict)
        model = build_model(config)
        model.load_state_dict(state_dict)
        return cls(model, config, device)

    def predict(
        self,
        input_data: np.ndarray | torch.Tensor,
        grid: np.ndarray | torch.Tensor | None = None,
    ) -> np.ndarray:
        """
        Run inference on input data.

        Args:
            input_data: Input tensor with shape depending on model type:
                - FNO1d: [B, Nx, initial_step * num_channels] or [Nx, initial_step * num_channels]
                - FNO2d: [B, Nx, Ny, initial_step * num_channels] or [Nx, Ny, initial_step * num_channels]
                - UNet1d: [B, in_channels, Nx] or [in_channels, Nx]
                - UNet2d: [B, in_channels, Nx, Ny] or [in_channels, Nx, Ny]
            grid: Optional grid coordinates. If None, will be auto-generated.
                - FNO1d: [B, Nx, 1]
                - FNO2d: [B, Nx, Ny, 2]

        Returns:
            Output prediction as numpy array
        """
        # Convert to tensor if needed
        if isinstance(input_data, np.ndarray):
            input_data = torch.from_numpy(input_data.astype(np.float32))

        # Add batch dimension if needed
        input_data = self._ensure_batch_dim(input_data)

        input_data = input_data.to(self.device)

        with torch.no_grad():
            if self.config.model_type in ("FNO1d", "FNO2d"):
                # FNO requires grid input
                if grid is None:
                    grid = self._generate_grid(input_data)
                elif isinstance(grid, np.ndarray):
                    grid = torch.from_numpy(grid.astype(np.float32))
                grid = grid.to(self.device)
                output = self.model(input_data, grid)
            else:
                # UNet doesn't need grid
                output = self.model(input_data)

        return output.cpu().numpy()

    def predict_autoregressive(
        self,
        initial_condition: np.ndarray | torch.Tensor,
        num_steps: int,
        grid: np.ndarray | torch.Tensor | None = None,
    ) -> np.ndarray:
        """
        Run autoregressive prediction for multiple time steps.

        This is the standard mode for time-dependent PDEs where the model
        predicts one step at a time, using its own output as input for the next step.

        Args:
            initial_condition: Initial state with shape:
                - FNO1d: [B, Nx, initial_step * num_channels]
                - FNO2d: [B, Nx, Ny, initial_step * num_channels]
                - UNet1d: [B, in_channels, Nx]
                - UNet2d: [B, in_channels, Nx, Ny]
            num_steps: Number of time steps to predict
            grid: Optional grid coordinates (for FNO models)

        Returns:
            Predictions for all time steps: [B, ..., num_steps, out_channels]
        """
        # Convert to tensor if needed
        if isinstance(initial_condition, np.ndarray):
            initial_condition = torch.from_numpy(initial_condition.astype(np.float32))

        initial_condition = self._ensure_batch_dim(initial_condition)
        initial_condition = initial_condition.to(self.device)

        if grid is None and self.config.model_type in ("FNO1d", "FNO2d"):
            grid = self._generate_grid(initial_condition)
        if grid is not None:
            if isinstance(grid, np.ndarray):
                grid = torch.from_numpy(grid.astype(np.float32))
            grid = grid.to(self.device)

        predictions = []
        current_input = initial_condition.clone()

        with torch.no_grad():
            for _ in range(num_steps):
                if self.config.model_type in ("FNO1d", "FNO2d"):
                    output = self.model(current_input, grid)
                else:
                    output = self.model(current_input)

                predictions.append(output.cpu())

                # Update input for next step (roll forward)
                current_input = self._roll_input(current_input, output)

        # Stack predictions along time dimension
        predictions = torch.cat(predictions, dim=-2)
        return predictions.numpy()

    def _ensure_batch_dim(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure input has batch dimension."""
        if self.config.model_type == "FNO1d":
            # Expected: [B, Nx, C]
            if x.dim() == 2:
                x = x.unsqueeze(0)
        elif self.config.model_type == "FNO2d":
            # Expected: [B, Nx, Ny, C]
            if x.dim() == 3:
                x = x.unsqueeze(0)
        elif self.config.model_type == "UNet1d":
            # Expected: [B, C, Nx]
            if x.dim() == 2:
                x = x.unsqueeze(0)
        elif self.config.model_type == "UNet2d":
            # Expected: [B, C, Nx, Ny]
            if x.dim() == 3:
                x = x.unsqueeze(0)
        return x

    def _generate_grid(self, x: torch.Tensor) -> torch.Tensor:
        """Generate normalized grid coordinates for FNO models."""
        batch_size = x.shape[0]

        if self.config.model_type == "FNO1d":
            # x: [B, Nx, C]
            nx = x.shape[1]
            grid_x = torch.linspace(0, 1, nx, device=x.device)
            grid = grid_x.view(1, nx, 1).expand(batch_size, -1, -1)

        elif self.config.model_type == "FNO2d":
            # x: [B, Nx, Ny, C]
            nx, ny = x.shape[1], x.shape[2]
            grid_x = torch.linspace(0, 1, nx, device=x.device)
            grid_y = torch.linspace(0, 1, ny, device=x.device)
            gx, gy = torch.meshgrid(grid_x, grid_y, indexing="ij")
            grid = torch.stack([gx, gy], dim=-1)
            grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1)

        else:
            raise ValueError(f"Grid not needed for {self.config.model_type}")

        return grid

    def _roll_input(self, current_input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """Roll input forward for autoregressive prediction."""
        if self.config.model_type == "FNO1d":
            # current_input: [B, Nx, initial_step * num_channels]
            # output: [B, Nx, 1, num_channels]
            nc = self.config.num_channels
            output_flat = output.squeeze(-2)  # [B, Nx, num_channels]
            # Roll: drop first time step, append new output
            new_input = torch.cat([
                current_input[..., nc:],  # Drop first step
                output_flat,              # Append new prediction
            ], dim=-1)
            return new_input

        elif self.config.model_type == "FNO2d":
            # current_input: [B, Nx, Ny, initial_step * num_channels]
            # output: [B, Nx, Ny, 1, num_channels]
            nc = self.config.num_channels
            output_flat = output.squeeze(-2)  # [B, Nx, Ny, num_channels]
            new_input = torch.cat([
                current_input[..., nc:],
                output_flat,
            ], dim=-1)
            return new_input

        elif self.config.model_type == "UNet1d":
            # current_input: [B, in_channels, Nx]
            # output: [B, out_channels, Nx]
            oc = self.config.out_channels
            new_input = torch.cat([
                current_input[:, oc:, :],  # Drop first channels
                output,                     # Append new prediction
            ], dim=1)
            return new_input

        elif self.config.model_type == "UNet2d":
            # current_input: [B, in_channels, Nx, Ny]
            # output: [B, out_channels, Nx, Ny]
            oc = self.config.out_channels
            new_input = torch.cat([
                current_input[:, oc:, :, :],
                output,
            ], dim=1)
            return new_input

        raise ValueError(f"Unknown model type: {self.config.model_type}")

    @property
    def input_shape_hint(self) -> str:
        """Get a hint about expected input shape."""
        if self.config.model_type == "FNO1d":
            return f"[B, Nx, {self.config.initial_step * self.config.num_channels}] (initial_step={self.config.initial_step})"
        elif self.config.model_type == "FNO2d":
            return f"[B, Nx, Ny, {self.config.initial_step * self.config.num_channels}] (initial_step={self.config.initial_step})"
        elif self.config.model_type == "UNet1d":
            return f"[B, {self.config.in_channels}, Nx]"
        elif self.config.model_type == "UNet2d":
            return f"[B, {self.config.in_channels}, Nx, Ny]"
        return "Unknown"

    def __repr__(self) -> str:
        return (
            f"PDEPredictor(\n"
            f"  model_type={self.config.model_type},\n"
            f"  input_shape={self.input_shape_hint},\n"
            f"  out_channels={self.config.out_channels},\n"
            f"  device={self.device}\n"
            f")"
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def predict_pde(
    checkpoint_path: str | Path,
    input_data: np.ndarray | torch.Tensor,
    grid: np.ndarray | torch.Tensor | None = None,
    device: str | None = None,
) -> np.ndarray:
    """
    One-shot prediction function.

    For repeated predictions, use PDEPredictor.load() to avoid reloading the model.

    Args:
        checkpoint_path: Path to the .pt checkpoint file
        input_data: Input tensor (see PDEPredictor.predict for shape details)
        grid: Optional grid coordinates (for FNO models)
        device: Device to run on (default: cuda if available)

    Returns:
        Output prediction as numpy array
    """
    predictor = PDEPredictor.load(checkpoint_path, device=device)
    return predictor.predict(input_data, grid=grid)


def list_available_models(directory: str | Path = ".") -> list[dict[str, Any]]:
    """
    List available PDEBench model checkpoints in a directory.

    Args:
        directory: Directory to search for .pt files

    Returns:
        List of dictionaries with model info
    """
    directory = Path(directory)
    models = []

    for pt_file in directory.glob("*.pt"):
        try:
            ckpt = torch.load(pt_file, map_location="cpu", weights_only=False)
            if "model_state_dict" in ckpt:
                config = infer_model_config(ckpt["model_state_dict"])
                models.append({
                    "path": str(pt_file),
                    "name": pt_file.stem,
                    "model_type": config.model_type,
                    "in_channels": config.in_channels,
                    "out_channels": config.out_channels,
                })
        except Exception:
            pass

    return models


# =============================================================================
# Example usage and testing
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PDEBench Inference Demo")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument("--list", action="store_true", help="List available models")
    args = parser.parse_args()

    if args.list:
        print("Available models:")
        for model_info in list_available_models():
            print(f"  {model_info['name']}: {model_info['model_type']}")

    elif args.checkpoint:
        # Demo: load model and show info
        predictor = PDEPredictor.load(args.checkpoint)
        print(predictor)
        print(f"\nExpected input shape: {predictor.input_shape_hint}")

        # Create dummy input for testing
        if predictor.config.model_type == "FNO1d":
            dummy = np.random.randn(1, 256, predictor.config.initial_step).astype(np.float32)
        elif predictor.config.model_type == "FNO2d":
            dummy = np.random.randn(1, 64, 64, predictor.config.initial_step).astype(np.float32)
        elif predictor.config.model_type == "UNet1d":
            dummy = np.random.randn(1, predictor.config.in_channels, 256).astype(np.float32)
        elif predictor.config.model_type == "UNet2d":
            dummy = np.random.randn(1, predictor.config.in_channels, 64, 64).astype(np.float32)
        else:
            dummy = None

        if dummy is not None:
            print(f"\nRunning inference with dummy input of shape {dummy.shape}...")
            output = predictor.predict(dummy)
            print(f"Output shape: {output.shape}")

    else:
        parser.print_help()
