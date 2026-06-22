"""ResNet blocks and custom layers for VAE architecture."""

import torch
import torch.nn as nn


class SqueezeExcitationBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.

    Applies adaptive channel-wise feature recalibration using:
    1. Squeeze: Global average pooling to get channel descriptors
    2. Excitation: Two FC layers with sigmoid to learn channel weights
    3. Scale: Multiply input features by learned weights

    Args:
        channels: Number of input channels.
        reduction: Reduction ratio for bottleneck FC layer (default: 8).
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        reduced_channels = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, reduced_channels)
        self.fc2 = nn.Linear(reduced_channels, channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention to input tensor."""
        batch_size, channels, _, _ = x.size()

        # Squeeze: Global average pooling
        scale = x.mean((2, 3), keepdim=True)  # (B, C, 1, 1)
        scale = scale.view(batch_size, channels)  # (B, C)

        # Excitation: FC layers with bottleneck
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.sigmoid(scale)
        scale = scale.view(batch_size, channels, 1, 1)  # (B, C, 1, 1)

        # Scale input by learned channel weights
        return x * scale


class ResidualBlock(nn.Module):
    """
    Residual block with two convolutional layers and a skip connection.

    Optionally includes Squeeze-and-Excitation channel attention.

    Args:
        channels: Number of input and output channels.
        use_se: Whether to use Squeeze-and-Excitation block
            (default: True).
        se_reduction: Reduction ratio for SE block (default: 8).
    """

    def __init__(
        self, channels: int, use_se: bool = True, se_reduction: int = 8
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

        # Optional Squeeze-and-Excitation block
        self.se = (
            SqueezeExcitationBlock(channels, se_reduction) if use_se else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connection and optional SE
        attention."""
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Apply SE block before skip connection if enabled
        if self.se is not None:
            out = self.se(out)

        out = out + identity
        out = self.relu(out)

        return out


class DownsampleBlock(nn.Module):
    """
    Downsampling block with two residual blocks and spatial
    downsampling.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # Downsampling with stride 2
        self.downsample = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1
        )
        self.relu = nn.ReLU()

        # Two residual blocks at the new resolution
        self.res_block1 = ResidualBlock(out_channels)
        self.res_block2 = ResidualBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with downsampling and residual blocks."""
        x = self.downsample(x)
        x = self.relu(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        return x


class UpsampleBlock(nn.Module):
    """
    Upsampling block with two residual blocks and spatial upsampling.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # Two residual blocks at current resolution
        self.res_block1 = ResidualBlock(in_channels)
        self.res_block2 = ResidualBlock(in_channels)
        self.relu = nn.ReLU()

        # Upsampling with transposed convolution
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual blocks and upsampling."""
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.upsample(x)
        x = self.relu(x)
        return x
