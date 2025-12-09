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


class PINN1d(nn.Module):
    """
    Physics-Informed Neural Network (PINN) for 1D time-dependent PDEs.

    This is a simple fully-connected neural network (FNN) that takes (x, t)
    coordinates as input and outputs the solution u(x, t).

    Architecture: [in_channels] + [hidden_dim] * (num_layers - 1) + [out_channels]
    where num_layers is the total number of linear layers.
    Activation: tanh (matching deepxde default)
    """

    def __init__(
        self,
        in_channels: int = 2,  # (x, t) coordinates
        out_channels: int = 1,
        hidden_dim: int = 40,
        num_layers: int = 7,  # Total number of linear layers (including input and output)
        activation: str = "tanh",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Build layers: num_layers total linear layers
        # Layer 0: in_channels -> hidden_dim
        # Layers 1 to num_layers-2: hidden_dim -> hidden_dim
        # Layer num_layers-1: hidden_dim -> out_channels
        layers = []
        # Input layer
        layers.append(nn.Linear(in_channels, hidden_dim))
        # Hidden layers (num_layers - 2 of them)
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        # Output layer
        layers.append(nn.Linear(hidden_dim, out_channels))

        self.linears = nn.ModuleList(layers)

        # Activation function
        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            self.activation = torch.tanh

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input coordinates [B, N, 2] or [B*N, 2] where last dim is (x, t)

        Returns:
            Solution values [B, N, out_channels] or [B*N, out_channels]
        """
        # Apply layers with activation (except last layer)
        for i, layer in enumerate(self.linears[:-1]):
            x = self.activation(layer(x))
        # Output layer (no activation, but apply ReLU for non-negative outputs like diffusion-sorption)
        x = self.linears[-1](x)
        return torch.relu(x)  # Ensure non-negative output for diffusion-sorption


# =============================================================================
# Model Configuration and Inference
# =============================================================================

ModelType = Literal["FNO1d", "FNO2d", "UNet1d", "UNet2d", "PINN1d"]


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
    # PINN specific
    hidden_dim: int | None = None
    num_layers: int | None = None


def infer_model_config(state_dict: dict[str, torch.Tensor]) -> ModelConfig:
    """Infer model architecture and parameters from checkpoint state dict."""
    keys = set(state_dict.keys())

    # Detect model type
    is_fno = "fc0.weight" in keys and "conv0.weights1" in keys
    is_unet = "encoder1.enc1conv1.weight" in keys
    is_pinn = "linears.0.weight" in keys and "linears.0.bias" in keys

    if is_pinn:
        # PINN model (FNN architecture from deepxde)
        # Structure: linears.0, linears.1, ..., linears.N
        # Count number of layers
        num_layers = sum(1 for k in keys if k.startswith("linears.") and k.endswith(".weight"))

        # Get dimensions from weights
        first_layer_w = state_dict["linears.0.weight"]  # [hidden_dim, in_channels]
        last_layer_w = state_dict[f"linears.{num_layers - 1}.weight"]  # [out_channels, hidden_dim]

        in_channels = first_layer_w.shape[1]
        hidden_dim = first_layer_w.shape[0]
        out_channels = last_layer_w.shape[0]

        return ModelConfig(
            model_type="PINN1d",
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

    elif is_fno:
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
    elif config.model_type == "PINN1d":
        return PINN1d(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
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
                - PINN1d: [N, 2] or [B, N, 2] where last dim is (x, t) coordinates
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
            elif self.config.model_type == "PINN1d":
                # PINN takes (x, t) coordinates directly
                output = self.model(input_data)
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
        elif self.config.model_type == "PINN1d":
            # Expected: [N, 2] where N is number of query points
            # PINN can handle arbitrary number of points, no batch dim needed
            pass
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
        elif self.config.model_type == "PINN1d":
            return f"[N, {self.config.in_channels}] where N is number of (x, t) query points"
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
# Dataset Loading for Batch Inference
# =============================================================================

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


@dataclass
class TaskConfig:
    """Configuration for a PDE task."""
    name: str
    dataset_file: str
    model_checkpoints: list[str]
    initial_step: int = 10
    reduced_resolution: int = 1
    reduced_resolution_t: int = 1


# Default task configurations
DEFAULT_TASKS = {
    "darcy": TaskConfig(
        name="2D Darcy Flow",
        dataset_file="2D_DarcyFlow_beta1.0_Train.hdf5",
        model_checkpoints=[
            "2D_DarcyFlow_beta1.0_Train_Unet_PF_1.pt",
            "2D_DarcyFlow_beta1.0_Train_FNO.pt",
        ],
        initial_step=1,
        reduced_resolution=2,
    ),
    "burgers": TaskConfig(
        name="1D Burgers",
        dataset_file="1D_Burgers_Sols_Nu1.0.hdf5",
        model_checkpoints=[
            "1D_Burgers_Sols_Nu1.0_Unet-PF-20.pt",
            "1D_Burgers_Sols_Nu1.0_FNO.pt",
        ],
        initial_step=10,
        reduced_resolution=4,
        reduced_resolution_t=5,
    ),
    "diffsorp": TaskConfig(
        name="1D Diff-Sorp",
        dataset_file="1D_diff-sorp_NA_NA.h5",
        model_checkpoints=[
            "1D_diff-sorp_NA_NA_Unet-PF-20.pt",
            "1D_diff-sorp_NA_NA_FNO.pt",
            "1D_diff-sorp_NA_NA_0001.h5_PINN.pt-15000.pt",
        ],
        initial_step=10,
        reduced_resolution=1,
        reduced_resolution_t=1,
    ),
}

# Small model configurations (for LLM integration)
# These use smaller spatial/temporal resolution for compact input/output
SMALL_MODEL_TASKS = {
    "darcy_small": TaskConfig(
        name="2D Darcy Flow (Small)",
        dataset_file="2D_DarcyFlow_beta1.0_Train.hdf5",
        model_checkpoints=[
            "2D_DarcyFlow_beta1.0_Train_Unet_small.pt",
            "2D_DarcyFlow_beta1.0_Train_FNO_small.pt",
        ],
        initial_step=1,
        reduced_resolution=4,  # 128 -> 32
    ),
    "burgers_small": TaskConfig(
        name="1D Burgers (Small)",
        dataset_file="1D_Burgers_Sols_Nu1.0.hdf5",
        model_checkpoints=[
            # "1D_Burgers_Sols_Nu1.0_Unet_small.pt",
            "1D_Burgers_Sols_Nu1.0_Unet_small-PF-10.pt",
            "1D_Burgers_Sols_Nu1.0_FNO_small.pt",
        ],
        initial_step=5,
        reduced_resolution=16,   # 1024 -> 64
        reduced_resolution_t=5,  # 201 -> ~40
    ),
    "diffsorp_small": TaskConfig(
        name="1D Diff-Sorp (Small)",
        dataset_file="1D_diff-sorp_NA_NA.h5",
        model_checkpoints=[
            "1D_diff-sorp_NA_NA_Unet_small.pt",
            "1D_diff-sorp_NA_NA_Unet_small-PF-10.pt",
            "1D_diff-sorp_NA_NA_FNO_small.pt",
        ],
        initial_step=5,
        reduced_resolution=16,   # 1024 -> 64
        reduced_resolution_t=2,  # 101 -> ~50
    ),
}

# Merge small model tasks into default tasks
ALL_TASKS = {**DEFAULT_TASKS, **SMALL_MODEL_TASKS}


class PDEDataset:
    """Base class for PDE dataset loading."""

    def __init__(
        self,
        file_path: str | Path,
        initial_step: int = 10,
        reduced_resolution: int = 1,
        reduced_resolution_t: int = 1,
    ):
        self.file_path = Path(file_path)
        self.initial_step = initial_step
        self.reduced_resolution = reduced_resolution
        self.reduced_resolution_t = reduced_resolution_t

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (input, target) for sample idx."""
        raise NotImplementedError

    def get_batch(self, indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """Get a batch of samples."""
        inputs, targets = [], []
        for idx in indices:
            inp, tgt = self[idx]
            inputs.append(inp)
            targets.append(tgt)
        return np.stack(inputs), np.stack(targets)


class DarcyDataset(PDEDataset):
    """Dataset for 2D Darcy Flow."""

    def __init__(self, file_path: str | Path, reduced_resolution: int = 2, **kwargs):
        super().__init__(file_path, reduced_resolution=reduced_resolution, **kwargs)
        self._file = None
        self._nu = None
        self._tensor = None
        self._load_data()

    def _load_data(self):
        self._file = h5py.File(self.file_path, "r")
        self._nu = self._file["nu"]  # (N, 128, 128) - input
        self._tensor = self._file["tensor"]  # (N, 1, 128, 128) - output
        self._n_samples = self._nu.shape[0]

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        r = self.reduced_resolution
        # Input: permeability field (nu) with downsampling
        nu = self._nu[idx, ::r, ::r]  # (Nx, Ny)
        # Target: pressure field with downsampling
        target = self._tensor[idx, 0, ::r, ::r]  # (Nx, Ny)
        return nu.astype(np.float32), target.astype(np.float32)

    def prepare_input_unet(self, nu: np.ndarray) -> np.ndarray:
        """Prepare input for UNet2d: [B, 1, Nx, Ny]."""
        if nu.ndim == 2:
            return nu[np.newaxis, np.newaxis, :, :]
        elif nu.ndim == 3:
            return nu[:, np.newaxis, :, :]
        return nu

    def prepare_input_fno(self, nu: np.ndarray) -> np.ndarray:
        """Prepare input for FNO2d: [B, Nx, Ny, 1]."""
        if nu.ndim == 2:
            return nu[np.newaxis, :, :, np.newaxis]
        elif nu.ndim == 3:
            return nu[:, :, :, np.newaxis]
        return nu

    def close(self):
        if self._file:
            self._file.close()


class BurgersDataset(PDEDataset):
    """Dataset for 1D Burgers equation."""

    def __init__(
        self,
        file_path: str | Path,
        initial_step: int = 10,
        reduced_resolution: int = 4,
        reduced_resolution_t: int = 5,
        **kwargs,
    ):
        super().__init__(
            file_path,
            initial_step=initial_step,
            reduced_resolution=reduced_resolution,
            reduced_resolution_t=reduced_resolution_t,
        )
        self._file = None
        self._tensor = None
        self._load_data()

    def _load_data(self):
        self._file = h5py.File(self.file_path, "r")
        self._tensor = self._file["tensor"]  # (N, 201, 1024)
        self._n_samples = self._tensor.shape[0]

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        r = self.reduced_resolution
        rt = self.reduced_resolution_t
        # Get full trajectory with downsampling
        traj = self._tensor[idx, ::rt, ::r]  # (T_down, Nx_down)
        # Input: first initial_step time steps
        input_data = traj[:self.initial_step, :]  # (initial_step, Nx)
        # Target: next time step
        target = traj[self.initial_step, :]  # (Nx,)
        return input_data.astype(np.float32), target.astype(np.float32)

    def get_full_trajectory(self, idx: int) -> np.ndarray:
        """Get the full downsampled trajectory for autoregressive evaluation."""
        r = self.reduced_resolution
        rt = self.reduced_resolution_t
        return self._tensor[idx, ::rt, ::r].astype(np.float32)

    def prepare_input_unet(self, data: np.ndarray) -> np.ndarray:
        """Prepare input for UNet1d: [B, T, Nx]."""
        if data.ndim == 2:
            return data[np.newaxis, :, :]
        return data

    def prepare_input_fno(self, data: np.ndarray) -> np.ndarray:
        """Prepare input for FNO1d: [B, Nx, T]."""
        if data.ndim == 2:
            return data.T[np.newaxis, :, :]  # (1, Nx, T)
        elif data.ndim == 3:
            return data.transpose(0, 2, 1)  # (B, Nx, T)
        return data

    def close(self):
        if self._file:
            self._file.close()


class DiffSorpDataset(PDEDataset):
    """Dataset for 1D Diffusion-Sorption equation."""

    def __init__(
        self,
        file_path: str | Path,
        initial_step: int = 10,
        reduced_resolution: int = 1,
        reduced_resolution_t: int = 1,
        **kwargs,
    ):
        super().__init__(
            file_path,
            initial_step=initial_step,
            reduced_resolution=reduced_resolution,
            reduced_resolution_t=reduced_resolution_t,
            **kwargs
        )
        self._file = None
        self._seeds = None
        self._load_data()

    def _load_data(self):
        self._file = h5py.File(self.file_path, "r")
        # Get all seed keys (0000, 0001, ...)
        self._seeds = sorted([k for k in self._file.keys() if k.isdigit()])
        self._n_samples = len(self._seeds)

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        seed = self._seeds[idx]
        r = self.reduced_resolution
        rt = self.reduced_resolution_t
        # data shape: (101, 1024, 1)
        data = self._file[seed]["data"][:]
        data = data.squeeze(-1)  # (101, 1024)
        # Apply downsampling
        data = data[::rt, ::r]  # (T_down, Nx_down)
        # Input: first initial_step time steps
        input_data = data[:self.initial_step, :]  # (initial_step, Nx)
        # Target: next time step
        target = data[self.initial_step, :]  # (Nx,)
        return input_data.astype(np.float32), target.astype(np.float32)

    def get_full_trajectory(self, idx: int) -> np.ndarray:
        """Get the full downsampled trajectory for autoregressive evaluation."""
        seed = self._seeds[idx]
        r = self.reduced_resolution
        rt = self.reduced_resolution_t
        data = self._file[seed]["data"][:]
        data = data.squeeze(-1)  # (101, 1024)
        return data[::rt, ::r].astype(np.float32)

    def prepare_input_unet(self, data: np.ndarray) -> np.ndarray:
        """Prepare input for UNet1d: [B, T, Nx]."""
        if data.ndim == 2:
            return data[np.newaxis, :, :]
        return data

    def prepare_input_fno(self, data: np.ndarray) -> np.ndarray:
        """Prepare input for FNO1d: [B, Nx, T]."""
        if data.ndim == 2:
            return data.T[np.newaxis, :, :]  # (1, Nx, T)
        elif data.ndim == 3:
            return data.transpose(0, 2, 1)  # (B, Nx, T)
        return data

    def get_pinn_coords_and_target(self, idx: int, t_idx: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Get (x, t) coordinates and target values for PINN evaluation.

        For Diff-Sorp: x in [0, 1], t in [0, 500]

        Args:
            idx: Sample index
            t_idx: Specific time step index (default: evaluate at t=initial_step)

        Returns:
            coords: [N, 2] array of (x, t) coordinates
            target: [N,] array of target values
        """
        seed = self._seeds[idx]
        data = self._file[seed]["data"][:]  # (101, 1024, 1)
        data = data.squeeze(-1)  # (101, 1024)

        # Get spatial and temporal grids
        n_t, n_x = data.shape
        x_coords = np.linspace(0, 1, n_x)
        t_max = 500.0  # Diff-Sorp has t in [0, 500]
        t_coords = np.linspace(0, t_max, n_t)

        if t_idx is None:
            t_idx = self.initial_step

        # Create (x, t) coordinate pairs for the specific time step
        t_val = t_coords[t_idx]
        coords = np.stack([x_coords, np.full(n_x, t_val)], axis=-1)  # [Nx, 2]
        target = data[t_idx, :]  # [Nx,]

        return coords.astype(np.float32), target.astype(np.float32)

    def get_pinn_full_coords_and_target(self, idx: int, n_last_time_steps: int = 20) -> tuple[np.ndarray, np.ndarray]:
        """
        Get (x, t) coordinates and target values for PINN evaluation over multiple time steps.

        Args:
            idx: Sample index
            n_last_time_steps: Number of last time steps to evaluate

        Returns:
            coords: [N, 2] array of (x, t) coordinates
            target: [N,] array of target values
        """
        seed = self._seeds[idx]
        data = self._file[seed]["data"][:]  # (101, 1024, 1)
        data = data.squeeze(-1)  # (101, 1024)

        # Get spatial and temporal grids
        n_t, n_x = data.shape
        x_coords = np.linspace(0, 1, n_x)
        t_max = 500.0
        t_coords = np.linspace(0, t_max, n_t)

        # Get last n time steps
        t_indices = range(n_t - n_last_time_steps, n_t)

        all_coords = []
        all_targets = []
        for t_idx in t_indices:
            t_val = t_coords[t_idx]
            coords = np.stack([x_coords, np.full(n_x, t_val)], axis=-1)
            all_coords.append(coords)
            all_targets.append(data[t_idx, :])

        coords = np.concatenate(all_coords, axis=0)  # [n_last_time_steps * Nx, 2]
        target = np.concatenate(all_targets, axis=0)  # [n_last_time_steps * Nx,]

        return coords.astype(np.float32), target.astype(np.float32)

    def close(self):
        if self._file:
            self._file.close()


# =============================================================================
# Batch Inference Runner
# =============================================================================

@dataclass
class InferenceResult:
    """Results from batch inference."""
    model_name: str
    task_name: str
    n_samples: int
    mse: float
    mae: float
    predictions: np.ndarray | None = None
    targets: np.ndarray | None = None


def run_batch_inference(
    predictor: PDEPredictor,
    dataset: PDEDataset,
    sample_indices: list[int] | None = None,
    batch_size: int = 32,
    save_predictions: bool = True,
    show_progress: bool = True,
) -> InferenceResult:
    """
    Run batch inference on a dataset.

    Args:
        predictor: PDEPredictor instance
        dataset: PDEDataset instance
        sample_indices: Indices of samples to evaluate (default: all)
        batch_size: Batch size for inference
        save_predictions: Whether to save all predictions
        show_progress: Whether to show progress bar

    Returns:
        InferenceResult with metrics and optionally predictions
    """
    if sample_indices is None:
        sample_indices = list(range(len(dataset)))

    n_samples = len(sample_indices)
    all_predictions = []
    all_targets = []
    total_mse = 0.0
    total_mae = 0.0

    # Determine input preparation function based on model type
    is_fno = predictor.config.model_type in ("FNO1d", "FNO2d")
    if isinstance(dataset, DarcyDataset):
        prepare_fn = dataset.prepare_input_fno if is_fno else dataset.prepare_input_unet
    else:
        prepare_fn = dataset.prepare_input_fno if is_fno else dataset.prepare_input_unet

    # Create progress iterator
    n_batches = (n_samples + batch_size - 1) // batch_size
    batch_iter = range(0, n_samples, batch_size)

    if show_progress:
        try:
            from tqdm import tqdm
            batch_iter = tqdm(batch_iter, total=n_batches, desc="Inference", unit="batch")
        except ImportError:
            print(f"Processing {n_samples} samples in {n_batches} batches...")

    for batch_start in batch_iter:
        batch_end = min(batch_start + batch_size, n_samples)
        batch_indices = sample_indices[batch_start:batch_end]

        # Get batch data
        inputs, targets = dataset.get_batch(batch_indices)

        # Prepare input for model
        model_input = prepare_fn(inputs)

        # Run inference
        predictions = predictor.predict(model_input)

        # Post-process predictions based on model type
        if predictor.config.model_type == "FNO2d":
            # FNO2d output: [B, Nx, Ny, 1, 1] -> [B, Nx, Ny]
            predictions = predictions.squeeze(-1).squeeze(-1)
        elif predictor.config.model_type == "UNet2d":
            # UNet2d output: [B, 1, Nx, Ny] -> [B, Nx, Ny]
            predictions = predictions.squeeze(1)
        elif predictor.config.model_type == "FNO1d":
            # FNO1d output: [B, Nx, 1, 1] -> [B, Nx]
            predictions = predictions.squeeze(-1).squeeze(-1)
        elif predictor.config.model_type == "UNet1d":
            # UNet1d output: [B, 1, Nx] -> [B, Nx]
            predictions = predictions.squeeze(1)

        # Calculate metrics
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        total_mse += mse * len(batch_indices)
        total_mae += mae * len(batch_indices)

        if save_predictions:
            all_predictions.append(predictions)
            all_targets.append(targets)

    # Aggregate results
    avg_mse = total_mse / n_samples
    avg_mae = total_mae / n_samples

    predictions_array = np.concatenate(all_predictions, axis=0) if save_predictions else None
    targets_array = np.concatenate(all_targets, axis=0) if save_predictions else None

    return InferenceResult(
        model_name=predictor.config.model_type,
        task_name=type(dataset).__name__.replace("Dataset", ""),
        n_samples=n_samples,
        mse=float(avg_mse),
        mae=float(avg_mae),
        predictions=predictions_array,
        targets=targets_array,
    )


def run_pinn_inference(
    predictor: PDEPredictor,
    dataset: DiffSorpDataset,
    sample_indices: list[int] | None = None,
    save_predictions: bool = True,
    show_progress: bool = True,
) -> InferenceResult:
    """
    Run PINN inference on DiffSorp dataset.

    PINN models take (x, t) coordinates directly as input, unlike FNO/UNet models
    that take time series data. This function handles the coordinate-based evaluation.

    Args:
        predictor: PDEPredictor instance with PINN model
        dataset: DiffSorpDataset instance
        sample_indices: Indices of samples to evaluate (default: all)
        save_predictions: Whether to save all predictions
        show_progress: Whether to show progress bar

    Returns:
        InferenceResult with metrics and optionally predictions
    """
    if sample_indices is None:
        sample_indices = list(range(len(dataset)))

    n_samples = len(sample_indices)
    all_predictions = []
    all_targets = []
    total_mse = 0.0
    total_mae = 0.0

    # Create progress iterator
    sample_iter = sample_indices
    if show_progress:
        try:
            from tqdm import tqdm
            sample_iter = tqdm(sample_indices, desc="PINN Inference", unit="sample")
        except ImportError:
            print(f"Processing {n_samples} samples...")

    for idx in sample_iter:
        # Get (x, t) coordinates and target for this sample
        # Evaluate at the initial_step time point (same as other models)
        coords, target = dataset.get_pinn_coords_and_target(idx)

        # Run inference
        predictions = predictor.predict(coords)  # [N, 1]
        predictions = predictions.squeeze(-1)  # [N,]

        # Calculate metrics
        mse = np.mean((predictions - target) ** 2)
        mae = np.mean(np.abs(predictions - target))
        total_mse += mse
        total_mae += mae

        if save_predictions:
            all_predictions.append(predictions)
            all_targets.append(target)

    # Aggregate results
    avg_mse = total_mse / n_samples
    avg_mae = total_mae / n_samples

    predictions_array = np.stack(all_predictions, axis=0) if save_predictions else None
    targets_array = np.stack(all_targets, axis=0) if save_predictions else None

    return InferenceResult(
        model_name=predictor.config.model_type,
        task_name=type(dataset).__name__.replace("Dataset", ""),
        n_samples=n_samples,
        mse=float(avg_mse),
        mae=float(avg_mae),
        predictions=predictions_array,
        targets=targets_array,
    )


def save_results(
    result: InferenceResult,
    output_dir: str | Path,
    model_name: str,
    save_predictions: bool = True,
) -> Path:
    """
    Save inference results to files.

    Args:
        result: InferenceResult instance
        output_dir: Output directory
        model_name: Model name for file naming
        save_predictions: Whether to save prediction arrays

    Returns:
        Path to the results directory
    """
    import json
    from datetime import datetime

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics = {
        "model_name": model_name,
        "task_name": result.task_name,
        "n_samples": result.n_samples,
        "mse": result.mse,
        "mae": result.mae,
        "timestamp": datetime.now().isoformat(),
    }

    metrics_file = output_dir / f"{model_name}_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save predictions and targets
    if save_predictions and result.predictions is not None:
        np.save(output_dir / f"{model_name}_predictions.npy", result.predictions)
        np.save(output_dir / f"{model_name}_targets.npy", result.targets)

    return output_dir


def run_all_models(
    data_dir: str | Path = ".",
    output_dir: str | Path = "inference_results",
    tasks: list[str] | None = None,
    models: list[str] | None = None,
    n_samples: int | None = None,
    batch_size: int = 32,
    save_predictions: bool = True,
    device: str | None = None,
    include_small: bool = True,
) -> dict[str, InferenceResult]:
    """
    Run inference on all available models and datasets.

    Args:
        data_dir: Directory containing datasets and model checkpoints
        output_dir: Directory to save results
        tasks: List of tasks to evaluate (default: all available)
            - Standard tasks: "darcy", "burgers", "diffsorp"
            - Small model tasks: "darcy_small", "burgers_small", "diffsorp_small"
        models: List of specific model files to use (default: all for each task)
        n_samples: Number of samples to evaluate (default: all)
        batch_size: Batch size for inference
        save_predictions: Whether to save prediction arrays
        device: Device to run on (default: auto)
        include_small: Whether to include small model tasks when tasks is None

    Returns:
        Dictionary mapping model names to InferenceResult
    """
    if not HAS_H5PY:
        raise ImportError("h5py is required for batch inference. Install with: pip install h5py")

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    # Select task configurations
    available_tasks = ALL_TASKS if include_small else DEFAULT_TASKS

    if tasks is None:
        tasks = list(available_tasks.keys())

    results = {}

    for task_name in tasks:
        if task_name not in available_tasks:
            print(f"Warning: Unknown task '{task_name}', skipping...")
            continue

        task_config = available_tasks[task_name]
        dataset_path = data_dir / task_config.dataset_file

        if not dataset_path.exists():
            print(f"Warning: Dataset not found: {dataset_path}, skipping task '{task_name}'...")
            continue

        print(f"\n{'='*60}")
        print(f"Task: {task_config.name}")
        print(f"Dataset: {dataset_path.name}")
        print(f"{'='*60}")

        # Load dataset based on task type (handle both standard and small tasks)
        base_task = task_name.replace("_small", "")  # Get base task name

        if base_task == "darcy":
            dataset = DarcyDataset(
                dataset_path,
                reduced_resolution=task_config.reduced_resolution,
            )
        elif base_task == "burgers":
            dataset = BurgersDataset(
                dataset_path,
                initial_step=task_config.initial_step,
                reduced_resolution=task_config.reduced_resolution,
                reduced_resolution_t=task_config.reduced_resolution_t,
            )
        elif base_task == "diffsorp":
            dataset = DiffSorpDataset(
                dataset_path,
                initial_step=task_config.initial_step,
                reduced_resolution=task_config.reduced_resolution,
                reduced_resolution_t=task_config.reduced_resolution_t,
            )
        else:
            print(f"Warning: Unknown base task '{base_task}', skipping...")
            continue

        # Determine sample indices
        total_samples = len(dataset)
        if n_samples is not None:
            sample_indices = list(range(min(n_samples, total_samples)))
        else:
            sample_indices = list(range(total_samples))

        print(f"Total samples: {total_samples}, Evaluating: {len(sample_indices)}")

        # Get model checkpoints to evaluate
        if models is not None:
            checkpoints = [m for m in models if any(m.endswith(c) or c in m for c in task_config.model_checkpoints)]
        else:
            checkpoints = task_config.model_checkpoints

        for ckpt_name in checkpoints:
            ckpt_path = data_dir / ckpt_name
            if not ckpt_path.exists():
                print(f"  Warning: Checkpoint not found: {ckpt_path}, skipping...")
                continue

            print(f"\n  Model: {ckpt_name}")
            print(f"  Loading model...")

            try:
                predictor = PDEPredictor.load(ckpt_path, device=device)
                print(f"  Model type: {predictor.config.model_type}")

                # Use different inference function for PINN models
                if predictor.config.model_type == "PINN1d" and isinstance(dataset, DiffSorpDataset):
                    result = run_pinn_inference(
                        predictor,
                        dataset,
                        sample_indices=sample_indices,
                        save_predictions=save_predictions,
                        show_progress=True,
                    )
                else:
                    result = run_batch_inference(
                        predictor,
                        dataset,
                        sample_indices=sample_indices,
                        batch_size=batch_size,
                        save_predictions=save_predictions,
                        show_progress=True,
                    )

                print(f"  MSE: {result.mse:.6e}")
                print(f"  MAE: {result.mae:.6e}")

                # Save results
                model_key = ckpt_path.stem
                save_results(result, output_dir / task_name, model_key, save_predictions)
                results[model_key] = result

            except Exception as e:
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()

        dataset.close()

    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Model':<45} {'MSE':<12} {'MAE':<12}")
    print("-" * 70)
    for model_name, result in results.items():
        print(f"{model_name:<45} {result.mse:<12.6e} {result.mae:<12.6e}")

    # Save summary
    import json
    summary = {
        name: {"mse": r.mse, "mae": r.mae, "n_samples": r.n_samples}
        for name, r in results.items()
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    return results


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="PDEBench Unified Inference Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available models
  python pdebench_inference.py --list

  # Run inference on all tasks with all models
  python pdebench_inference.py --run-all

  # Run inference on specific task
  python pdebench_inference.py --run-all --tasks darcy

  # Run with limited samples
  python pdebench_inference.py --run-all --n-samples 100

  # Specify output directory
  python pdebench_inference.py --run-all --output results/

  # Load and test a single model
  python pdebench_inference.py --checkpoint 2D_DarcyFlow_beta1.0_Train_Unet_PF_1.pt
""",
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--list", action="store_true", help="List available models")
    mode_group.add_argument("--run-all", action="store_true", help="Run batch inference on all models")
    mode_group.add_argument("--checkpoint", type=str, help="Load a single checkpoint for testing")

    # Batch inference options
    parser.add_argument("--tasks", nargs="+",
                        choices=["darcy", "burgers", "diffsorp",
                                 "darcy_small", "burgers_small", "diffsorp_small"],
                        help="Tasks to evaluate (default: all). Use *_small for small models.")
    parser.add_argument("--models", nargs="+", help="Specific model checkpoint files to use")
    parser.add_argument("--n-samples", type=int, help="Number of samples to evaluate (default: all)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--output", type=str, default="inference_results", help="Output directory")
    parser.add_argument("--data-dir", type=str, default=".", help="Data directory")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], help="Device to use")
    parser.add_argument("--no-save-predictions", action="store_true",
                        help="Don't save prediction arrays (only metrics)")
    parser.add_argument("--small-only", action="store_true",
                        help="Only evaluate small models (for LLM integration)")
    parser.add_argument("--original-only", action="store_true",
                        help="Only evaluate original models (exclude small models)")

    args = parser.parse_args()

    if args.list:
        print("Available models:")
        for model_info in list_available_models(args.data_dir):
            print(f"  {model_info['name']}: {model_info['model_type']}")

        print("\nAvailable tasks (original models):")
        for task_name, task_config in DEFAULT_TASKS.items():
            dataset_path = Path(args.data_dir) / task_config.dataset_file
            status = "found" if dataset_path.exists() else "NOT FOUND"
            print(f"  {task_name}: {task_config.name}")
            print(f"    Dataset: {task_config.dataset_file} [{status}]")
            print(f"    Resolution: {task_config.reduced_resolution}x (spatial)")
            print(f"    Models: {', '.join(task_config.model_checkpoints)}")

        print("\nAvailable tasks (small models for LLM):")
        for task_name, task_config in SMALL_MODEL_TASKS.items():
            dataset_path = Path(args.data_dir) / task_config.dataset_file
            status = "found" if dataset_path.exists() else "NOT FOUND"
            print(f"  {task_name}: {task_config.name}")
            print(f"    Dataset: {task_config.dataset_file} [{status}]")
            print(f"    Resolution: {task_config.reduced_resolution}x (spatial)")
            print(f"    Models: {', '.join(task_config.model_checkpoints)}")

    elif args.run_all:
        # Determine which tasks to include based on flags
        if args.small_only:
            tasks = args.tasks or list(SMALL_MODEL_TASKS.keys())
            include_small = True
        elif args.original_only:
            tasks = args.tasks or list(DEFAULT_TASKS.keys())
            include_small = False
        else:
            tasks = args.tasks
            include_small = True

        results = run_all_models(
            data_dir=args.data_dir,
            output_dir=args.output,
            tasks=tasks,
            models=args.models,
            n_samples=args.n_samples,
            batch_size=args.batch_size,
            save_predictions=not args.no_save_predictions,
            device=args.device,
            include_small=include_small,
        )

    elif args.checkpoint:
        # Demo: load model and show info
        predictor = PDEPredictor.load(args.checkpoint, device=args.device)
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
