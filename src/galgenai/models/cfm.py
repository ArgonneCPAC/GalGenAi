import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
from typing import Optional, Tuple


class ResBlock(nn.Module):
    """
    Residual block with time conditioning via adaptive group norm.

    The time embedding modulates the features through scale and shift
    parameters after group normalization. This is the standard approach
    in diffusion/flow models (e.g., ADM, DiT).

    Architecture:
        x -> GroupNorm -> scale/shift by time -> SiLU -> Conv
          -> GroupNorm -> scale/shift by time -> SiLU -> Dropout
          -> Conv -> + x
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        dropout: float = 0.1,
        num_groups: int = 8,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # First conv block
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        # Second conv block
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.dropout = nn.Dropout(dropout)

        # Conditioning: project to scale and shift for both norms
        # First norm uses in_channels, second uses out_channels
        # Output: 2*in_ch (scale1, shift1) + 2*out_ch (scale2, shift2)
        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, in_channels * 2 + out_channels * 2),
        )

        # Skip connection (identity if channels match, else 1x1 conv)
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, H, W) input features
            cond: (batch, cond_dim) time conditioning vector
        Returns:
            (batch, out_channels, H, W) output features
        """
        # Get conditioning parameters
        # Split: 2*in_ch for norm1, 2*out_ch for norm2
        cond_params = self.cond_proj(cond)
        params1, params2 = cond_params.split(
            [self.in_channels * 2, self.out_channels * 2], dim=1
        )
        scale1, shift1 = params1.chunk(2, dim=1)
        scale2, shift2 = params2.chunk(2, dim=1)

        # Reshape: (batch, channels) -> (batch, channels, 1, 1)
        scale1 = scale1[:, :, None, None]
        shift1 = shift1[:, :, None, None]
        scale2 = scale2[:, :, None, None]
        shift2 = shift2[:, :, None, None]

        # First block with adaptive normalization
        # Order: norm → scale/shift → activation → conv (standard AdaGN)
        h = self.norm1(x)
        h = h * (1 + scale1) + shift1
        h = F.silu(h)
        h = self.conv1(h)

        # Second block (same ordering)
        h = self.norm2(h)
        h = h * (1 + scale2) + shift2
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """
    Self-attention block for capturing long-range spatial
    dependencies.

    Applied at lower resolutions (e.g., 8x8, 16x16) where the
    computational cost is manageable. Uses multi-head attention with QKV
    projection.
    """

    def __init__(self, channels: int, num_heads: int = 4, num_groups: int = 8):
        super().__init__()

        if channels % num_heads != 0:
            raise ValueError(
                f"channels={channels} must be divisible by "
                f"num_heads={num_heads}"
            )

        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim**-0.5

        self.norm = nn.GroupNorm(num_groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(
        self, x: torch.Tensor, cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, H, W)
            cond: ignored, for interface compatibility with ResBlock
        Returns:
            (batch, channels, H, W)
        """
        del cond  # Unused, for interface compatibility
        B, C, H, W = x.shape

        h = self.norm(x)
        qkv = self.qkv(h)  # (B, 3*C, H, W)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, B, heads, H*W, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention: softmax(QK^T / sqrt(d)) V
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # (B, heads, H*W, head_dim)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)

        return x + self.proj(out)


# =====================================================================
# U-Net Velocity Field Network
# =====================================================================


class VelocityUNet(nn.Module):
    """
    U-Net architecture for predicting velocity field v(x_t, f, t).

    The network takes:
    - x_t: noisy image at time t (interpolation between noise and data)
    - f: conditional vector (can be a latent space projection of an
         image, FM embeddings, physical parameters, etc)
    - t: scalar time in [0, 1]

    And outputs the predicted velocity field (same shape as x_t).

    Architecture follows standard U-Net with:
    - Encoder: progressively downsamples while increasing channels
    - Middle: self-attention at lowest resolution
    - Decoder: progressively upsamples with skip connections from
      encoder
    - Conditioning: per-scalar sinusoidal embedding of cond_vec, summed
      with the time embedding to form a joint vector that modulates
      every ResBlock via AdaGN (DiT/SiT/SD3 pattern).

    Hyperparameter choices (following Samaddar et al. for scientific
    data):
    - Base channels: 64 (reduced from 128 since our images are 64x64,
      not 256x256)
    - Channel multipliers: [1, 2, 4, 4] gives [64, 128, 256, 256]
      channels
    - Attention at 16x16 and 8x8 resolutions
    - 2 residual blocks per resolution level
    """

    def __init__(
        self,
        in_channels: int = 5,
        cond_vec_dim: int = 32,
        input_size: int = 64,
        base_channels: int = 64,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (16, 8),
        dropout: float = 0.1,
        num_heads: int = 4,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.input_size = input_size
        self.base_channels = base_channels

        # Validate input_size is divisible by downsampling factor
        num_downsamples = len(channel_mult) - 1
        downsample_factor = 2**num_downsamples
        if input_size % downsample_factor != 0:
            raise ValueError(
                f"input_size={input_size} must be divisible by "
                f"{downsample_factor} for {num_downsamples} downsample stages"
            )

        # Compute channel counts at each level
        channels = [base_channels * m for m in channel_mult]

        # Time embedding: scalar -> vector
        time_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # f embedding: per-scalar sinusoidal Fourier features + MLP,
        # summed with the time embedding to form a joint vector that
        # modulates every ResBlock via AdaGN. The Fourier step is
        # applied inline in forward(); this MLP mirrors `time_mlp`.
        self.f_scalar_emb_dim = 32
        self.f_mlp = nn.Sequential(
            nn.Linear(cond_vec_dim * self.f_scalar_emb_dim, 4 * time_dim),
            nn.SiLU(),
            nn.Linear(4 * time_dim, time_dim),
        )

        # AdaGN conditioning dim = joint (time + f) embedding dim
        cond_dim = time_dim

        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Encoder (downsampling path)
        self.encoder_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        current_res = input_size  # Starting resolution
        prev_channels = base_channels

        for level, ch in enumerate(channels):
            level_blocks = nn.ModuleList()

            for _ in range(num_res_blocks):
                level_blocks.append(
                    ResBlock(prev_channels, ch, cond_dim, dropout)
                )
                prev_channels = ch

                # Add attention at specified resolutions
                if current_res in attention_resolutions:
                    level_blocks.append(AttentionBlock(ch, num_heads))

            self.encoder_blocks.append(level_blocks)

            # Downsample except at last level
            if level < len(channels) - 1:
                self.downsamplers.append(
                    nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)
                )
                current_res //= 2
            else:
                self.downsamplers.append(nn.Identity())

        # Middle block (at lowest resolution)
        self.middle = nn.ModuleList(
            [
                ResBlock(channels[-1], channels[-1], cond_dim, dropout),
                AttentionBlock(channels[-1], num_heads),
                ResBlock(channels[-1], channels[-1], cond_dim, dropout),
            ]
        )

        # Decoder (upsampling path)
        self.decoder_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        reversed_channels = list(reversed(channels))

        for level, ch in enumerate(reversed_channels):
            level_blocks = nn.ModuleList()

            # Output channels for this level
            out_ch = reversed_channels[
                min(level + 1, len(reversed_channels) - 1)
            ]
            if level == len(reversed_channels) - 1:
                out_ch = base_channels

            for i in range(
                num_res_blocks + 1
            ):  # +1 for skip connection processing
                # Input channels include skip connection
                block_in = prev_channels + ch if i == 0 else out_ch
                block_out = out_ch

                level_blocks.append(
                    ResBlock(block_in, block_out, cond_dim, dropout)
                )
                prev_channels = block_out

                if current_res in attention_resolutions:
                    level_blocks.append(AttentionBlock(block_out, num_heads))

            self.decoder_blocks.append(level_blocks)

            # Upsample except at last level
            if level < len(reversed_channels) - 1:
                self.upsamplers.append(
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
                )
                current_res *= 2
            else:
                self.upsamplers.append(nn.Identity())

        # Final output
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, in_channels, 3, padding=1),
        )

        # Initialize output conv to zero for stable training start
        nn.init.zeros_(self.conv_out[-1].weight)
        nn.init.zeros_(self.conv_out[-1].bias)

    @staticmethod
    def _sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Sinusoidal positional embedding for scalar time t ∈ [0, 1].

        Returns tensor of shape (batch, dim) with interleaved sin/cos
        terms.
        """
        device = t.device
        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half_dim, device=device) / half_dim
        )
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(
        self, x: torch.Tensor, f: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict velocity field at time t.

        Args:
            x: (batch, in_channels, H, W) noisy image x_t
            f: (batch, cond_vec_dim) conditioning vector
            t: (batch,) time values in [0, 1]

        Returns:
            (batch, in_channels, H, W) predicted velocity
        """
        # Validate input size matches expected size
        if x.shape[2] != self.input_size or x.shape[3] != self.input_size:
            raise ValueError(
                f"Expected input size {self.input_size}x{self.input_size}, "
                f"got {x.shape[2]}x{x.shape[3]}"
            )

        # Joint embedding e = t_emb + f_emb, injected into every
        # ResBlock via AdaGN (DiT/SiT/SD3 pattern).
        t_emb = self.time_mlp(
            self._sinusoidal_embedding(t, self.base_channels)
        )
        # Per-scalar sinusoidal Fourier features for f, vectorized:
        # flatten (B, f_dim) -> (B*f_dim,), embed, reshape back.
        f_sin = self._sinusoidal_embedding(
            f.reshape(-1), self.f_scalar_emb_dim
        ).reshape(x.shape[0], -1)
        f_emb = self.f_mlp(f_sin)
        cond = t_emb + f_emb

        # Initial conv
        h = self.conv_in(x)

        # Encoder - store outputs for skip connections
        skips = []
        for level_blocks, downsample in zip(
            self.encoder_blocks, self.downsamplers, strict=True
        ):
            for block in level_blocks:
                h = block(h, cond)
            skips.append(h)
            h = downsample(h)

        # Middle
        for block in self.middle:
            h = block(h, cond)

        # Decoder with skip connections
        for level_blocks, upsample in zip(
            self.decoder_blocks, self.upsamplers, strict=True
        ):
            # Concatenate skip connection
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)

            for block in level_blocks:
                h = block(h, cond)

            if not isinstance(upsample, nn.Identity):
                h = F.interpolate(h, scale_factor=2, mode="nearest")
                h = upsample(h)

        return self.conv_out(h)


# =====================================================================
# Full CFM Model
# =====================================================================


class CFM(nn.Module):
    """
    Conditional Flow Matching model.
    """

    def __init__(
        self,
        cond_vec_dim: int = 32,
        in_channels: int = 5,
        input_size: int = 64,
        base_channels: int = 64,
        **unet_kwargs,
    ):
        super().__init__()

        self.cond_vec_dim = cond_vec_dim
        self.in_channels = in_channels
        self.input_size = input_size

        # Trainable components
        self.velocity_net = VelocityUNet(
            in_channels=in_channels,
            cond_vec_dim=cond_vec_dim,
            input_size=input_size,
            base_channels=base_channels,
            **unet_kwargs,
        )

    def compute_loss(
        self,
        x1: torch.Tensor,
        f: torch.Tensor,
        ivar: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute CFM training loss.

        Args:
            x1: (batch, channels, H, W) real images from dataset
            f: (batch, cond_vec_dim) conditioning vector

        Returns:
            loss: scalar loss value
        """
        batch_size = x1.shape[0]
        device = x1.device

        # Sample noise (source distribution)
        x0 = torch.randn_like(x1)

        # Sample time uniformly
        t = torch.rand(batch_size, device=device)

        # Interpolate: x_t = (1-t)*x_0 + t*x_1
        t_broadcast = t[:, None, None, None]
        x_t = (1 - t_broadcast) * x0 + t_broadcast * x1

        # Target velocity (constant along the linear path)
        u_t = x1 - x0

        # Predict velocity
        v_pred = self.velocity_net(x_t, f, t)

        # Flow matching loss (weighted MSE when ivar/mask provided)
        if ivar is not None and mask is not None:
            squared_error = (v_pred - u_t).pow(2)
            mask_float = mask.float()
            weighted_error = squared_error * ivar * mask_float
            num_valid = mask_float.sum().clamp(min=1.0)
            loss = weighted_error.sum() / num_valid
        else:
            loss = F.mse_loss(v_pred, u_t)

        return loss

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        device,
        f: torch.Tensor,
        num_steps: int = 50,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """
        Generate samples using the trained model.

        Args:
            batch_size: number of samples to draw
            device: torch device for sampling
            f: (batch, cond_vec_dim) conditional features vector
            num_steps: number of ODE integration steps (Euler method)
            return_trajectory: if True, return intermediate states

        Returns:
            x1: (batch, channels, H, W) generated samples
            (optional) trajectory: list of intermediate states
        """
        self.eval()

        # Start from noise
        x = torch.randn(
            batch_size,
            self.in_channels,
            self.input_size,
            self.input_size,
            device=device,
        )

        trajectory = [x.clone()] if return_trajectory else None

        # Euler integration from t=0 to t=1
        dt = 1.0 / num_steps
        for step in range(num_steps):
            t = torch.full((batch_size,), step * dt, device=device)
            v = self.velocity_net(x, f, t)
            x = x + v * dt

            if return_trajectory:
                trajectory.append(x.clone())

        if return_trajectory:
            return x, trajectory
        return x

    @torch.no_grad()
    def sample_with_ode_solver(
        self,
        f: torch.Tensor,
        solver: str = "dopri5",
        rtol: float = 1e-5,
        atol: float = 1e-5,
    ) -> torch.Tensor:
        """
        Generate samples using adaptive ODE solver (requires
        torchdiffeq).

        This is more accurate than fixed-step Euler but slower.

        Args:
            f: (batch, cond_vec_dim) conditional features vector
            solver: ODE solver ('dopri5', 'rk4', etc.)
            rtol, atol: tolerances for adaptive solver

        Returns:
            Generated samples
        """

        self.eval()

        batch_size = f.shape[0]
        device = f.device

        def ode_fn(t, x):
            t_batch = torch.full((batch_size,), t.item(), device=device)
            return self.velocity_net(x, f, t_batch)

        # Initial condition
        x0 = torch.randn(
            batch_size,
            self.in_channels,
            self.input_size,
            self.input_size,
            device=device,
        )

        # Solve ODE
        t_span = torch.tensor([0.0, 1.0], device=device)
        solution = odeint(
            ode_fn, x0, t_span, method=solver, rtol=rtol, atol=atol
        )

        return solution[-1]  # Return final state at t=1


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
