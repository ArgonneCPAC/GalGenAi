"""Variational Autoencoder with ResNet architecture."""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional

from .layers import DownsampleBlock, UpsampleBlock

# Default channel progression for the VAE architecture
_DEFAULT_CHANNEL_DEPTHS = [16, 32, 64, 128, 256, 512, 512]


def _get_num_stages(input_size: int) -> int:
    """
    Determine number of stages based on input size.

    This prevents spatial dimensions from becoming too small during
    downsampling (which would cause BatchNorm errors with size 1x1).

    Args:
        input_size: Spatial size of input images (assumes square).

    Returns:
        Number of stages (channel depth levels) to use.
    """
    if input_size >= 128:
        return 7
    elif input_size >= 64:
        return 6
    else:
        return 5


class VAEEncoder(nn.Module):
    """
    VAE Encoder with configurable stages of downsampling using ResNet
    blocks.

    Args:
        in_channels: Number of input channels (e.g., 1 for grayscale).
        latent_dim: Dimension of the latent space.
        input_size: Spatial size of input images (assumes square
            images).
        channel_depths: List of channel depths for each downsampling
            stage.
        logvar_clamp: Tuple (min, max) to clamp logvar for numerical
            stability. Prevents overflow in exp() during
            reparameterization. Default: (-10.0, 10.0).
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 16,
        input_size: int = 32,
        channel_depths: Optional[List[int]] = None,
        logvar_clamp: Tuple[float, float] = (-10.0, 10.0),
    ):
        super().__init__()
        if channel_depths is None:
            num_stages = _get_num_stages(input_size)
            channel_depths = _DEFAULT_CHANNEL_DEPTHS[:num_stages]

        self.latent_dim = latent_dim
        self.input_size = input_size
        self.logvar_clamp = logvar_clamp

        # Initial channel-wise dense embedding
        self.initial_conv = nn.Conv2d(
            in_channels, channel_depths[0], kernel_size=3, padding=1
        )
        self.relu = nn.ReLU(inplace=True)

        # Downsampling stages
        self.downsample_blocks = nn.ModuleList()
        for i in range(len(channel_depths) - 1):
            self.downsample_blocks.append(
                DownsampleBlock(channel_depths[i], channel_depths[i + 1])
            )

        # Analytic flattened size (no dummy forward/BatchNorm side
        # effects)
        num_downsamples = len(channel_depths) - 1
        spatial = input_size // (2**num_downsamples)
        if spatial < 1:
            raise ValueError(
                f"Input size {input_size} too small for {num_downsamples} "
                "downsampling stages."
            )
        self.flatten_size = channel_depths[-1] * spatial * spatial

        # Dense layers for mean and log variance
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)

    def forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through convolutional layers."""
        x = self.initial_conv(x)
        x = self.relu(x)
        for block in self.downsample_blocks:
            x = block(x)
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height,
                width).

        Returns:
            Tuple of (mu, log_var) for the latent distribution.
        """
        x = self.forward_conv(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        # Clamp logvar for numerical stability
        # This prevents overflow in exp() during reparameterization
        logvar = torch.clamp(
            logvar, min=self.logvar_clamp[0], max=self.logvar_clamp[1]
        )

        return mu, logvar


class VAEDecoder(nn.Module):
    """
    VAE Decoder that mirrors the encoder architecture.

    Args:
        latent_dim: Dimension of the latent space.
        out_channels: Number of output channels (e.g., 1 for grayscale).
        input_size: Spatial size of target output images.
        channel_depths: List of channel depths for each upsampling stage
        (should be reverse of encoder's).
    """

    def __init__(
        self,
        latent_dim: int = 16,
        out_channels: int = 1,
        input_size: int = 32,
        channel_depths: Optional[List[int]] = None,
    ):
        super().__init__()
        if channel_depths is None:
            num_stages = _get_num_stages(input_size)
            channel_depths = _DEFAULT_CHANNEL_DEPTHS[:num_stages][::-1]

        self.latent_dim = latent_dim
        self.input_size = input_size
        self.channel_depths = channel_depths

        # Analytic unflatten size (mirror of encoder downsampling)
        num_upsamples = len(channel_depths) - 1
        spatial = input_size // (2**num_upsamples)
        if spatial < 1:
            raise ValueError(
                f"Input size {input_size} too small for {num_upsamples} "
                "upsampling stages."
            )
        self.unflatten_size = (spatial, spatial)
        self.flatten_size = channel_depths[0] * spatial * spatial

        # Dense layer to expand from latent space
        self.fc = nn.Linear(latent_dim, self.flatten_size)

        # Upsampling stages
        self.upsample_blocks = nn.ModuleList()
        for i in range(len(channel_depths) - 1):
            self.upsample_blocks.append(
                UpsampleBlock(channel_depths[i], channel_depths[i + 1])
            )

        # Final convolution to output channels
        self.final_conv = nn.Conv2d(
            channel_depths[-1], out_channels, kernel_size=3, padding=1
        )

        # Output activation: ReLU
        self.out_activation = nn.ReLU()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder.

        Args:
            z: Latent vector of shape (batch_size, latent_dim).

        Returns:
            Reconstructed image of shape (batch_size, out_channels, h,
                w).
        """
        x = self.fc(z)
        x = x.view(-1, self.channel_depths[0], *self.unflatten_size)

        for block in self.upsample_blocks:
            x = block(x)

        x = self.final_conv(x)
        x = self.out_activation(x)

        if x.shape[-2:] != (self.input_size, self.input_size):
            x = nn.functional.interpolate(
                x,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )

        return x


class VAE(nn.Module):
    """
    Complete Variational Autoencoder with ResNet architecture.

    Args:
        in_channels: Number of input channels.
        latent_dim: Dimension of the latent space.
        input_size: Spatial size of input images (assumes square).
        channel_depths: List of channel depths for the encoder. The
            decoder will use the reverse.
        logvar_clamp: Tuple (min, max) to clamp logvar in encoder for
            numerical stability. Prevents overflow during
            reparameterization. Default: (-10.0, 10.0).
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 16,
        input_size: int = 32,
        channel_depths: Optional[List[int]] = None,
        logvar_clamp: Tuple[float, float] = (-10.0, 10.0),
    ):
        super().__init__()
        if channel_depths is None:
            channel_depths = _DEFAULT_CHANNEL_DEPTHS

        # Adjust channel_depths based on input size
        num_stages = _get_num_stages(input_size)
        adjusted_channel_depths = channel_depths[:num_stages]

        self.encoder = VAEEncoder(
            in_channels,
            latent_dim,
            input_size,
            adjusted_channel_depths,
            logvar_clamp=logvar_clamp,
        )
        self.decoder = VAEDecoder(
            latent_dim,
            in_channels,
            input_size,
            channel_depths=[d for d in reversed(adjusted_channel_depths)],
        )

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + sigma * epsilon.

        Args:
            mu: Mean of the latent distribution.
            logvar: Log variance of the latent distribution.

        Returns:
            Sampled latent vector.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VAE.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height,
                width).

        Returns:
            Tuple of (reconstruction, mu, logvar).
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar

    def generate(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Generate new samples from the prior distribution.

        Args:
            num_samples: Number of samples to generate.
            device: Device to generate samples on.

        Returns:
            Generated samples.
        """
        z = torch.randn(num_samples, self.encoder.latent_dim, device=device)
        samples = self.decoder(z)
        return samples
