"""
AutoEncoder model for image tokenization in astromodal.

This module implements an AION-inspired autoencoder architecture with spatial latents
for encoding astronomical images into a compact latent representation.

Classes
-------
ResBlock
    Residual block with two convolutional layers
Encoder
    Encoder network that compresses images to latent space
Decoder
    Decoder network that reconstructs images from latent space
AutoEncoder
    Complete autoencoder with encoder and decoder
"""

from typing import Tuple
import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """
    Residual block with two 3x3 convolutions and ReLU activation.

    Parameters
    ----------
    channels : int
        Number of input and output channels

    Examples
    --------
    >>> block = ResBlock(64)
    >>> x = torch.randn(4, 64, 32, 32)
    >>> out = block(x)
    >>> out.shape
    torch.Size([4, 64, 32, 32])
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through residual block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, C, H, W)
        """
        residual = x
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        out = self.act(out + residual)
        return out


class Encoder(nn.Module):
    """
    Encoder network for compressing images to spatial latent representation.

    The encoder progressively downsamples the input image through multiple
    stages with residual blocks and strided convolutions.

    Parameters
    ----------
    in_channels : int
        Number of input image channels
    latent_dim : int
        Dimension of the latent space per spatial position
    proj_channels : int, default=64
        Number of channels after initial projection

    Attributes
    ----------
    latent_H : int
        Height of latent spatial representation
    latent_W : int
        Width of latent spatial representation

    Examples
    --------
    >>> encoder = Encoder(in_channels=12, latent_dim=2)
    >>> x = torch.randn(4, 12, 96, 96)
    >>> z = encoder(x)
    >>> z.shape
    torch.Size([4, 2, 24, 24])
    """

    def __init__(self, in_channels: int, latent_dim: int, proj_channels: int = 64):
        super().__init__()
        self.proj_channels = proj_channels

        # Initial 1x1 projection
        self.proj = nn.Conv2d(in_channels, proj_channels, kernel_size=1)

        # Encoder blocks with downsampling
        # Stage 1: 96 -> 48
        self.enc_block1 = nn.Sequential(
            ResBlock(proj_channels),
            nn.Conv2d(proj_channels, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        # Stage 2: 48 -> 24
        self.enc_block2 = nn.Sequential(
            ResBlock(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            ResBlock(128),
        )

        # Project to latent dimension
        self.to_latent = nn.Conv2d(128, latent_dim, kernel_size=1)

        # Store latent spatial dimensions (assuming 96x96 input)
        self.latent_H = 24  # 96 / 4
        self.latent_W = 24  # 96 / 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input image to latent representation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W)

        Returns
        -------
        torch.Tensor
            Latent tensor of shape (B, latent_dim, H/4, W/4)
        """
        h = self.proj(x)
        h = self.enc_block1(h)  # -> (B, 128, H/2, W/2)
        h = self.enc_block2(h)  # -> (B, 128, H/4, W/4)
        z_map = self.to_latent(h)
        return z_map


class Decoder(nn.Module):
    """
    Decoder network for reconstructing images from spatial latent representation.

    The decoder progressively upsamples the latent representation through multiple
    stages with residual blocks and transposed convolutions.

    Parameters
    ----------
    latent_dim : int
        Dimension of the latent space per spatial position
    out_channels : int
        Number of output image channels
    proj_channels : int, default=64
        Number of channels in intermediate layers

    Examples
    --------
    >>> decoder = Decoder(latent_dim=2, out_channels=12)
    >>> z = torch.randn(4, 2, 24, 24)
    >>> x_hat = decoder(z)
    >>> x_hat.shape
    torch.Size([4, 12, 96, 96])
    """

    def __init__(self, latent_dim: int, out_channels: int, proj_channels: int = 64):
        super().__init__()
        self.proj_channels = proj_channels

        # Project from latent dimension
        self.from_latent = nn.Conv2d(latent_dim, 128, kernel_size=1)

        # Decoder blocks with upsampling
        # Stage 1: 24 -> 48
        self.dec_block1 = nn.Sequential(
            ResBlock(128),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        # Stage 2: 48 -> 96
        self.dec_block2 = nn.Sequential(
            ResBlock(64),
            nn.ConvTranspose2d(64, proj_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            ResBlock(proj_channels),
        )

        # Final output projection
        self.out_conv = nn.Conv2d(proj_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, z_map: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to image.

        Parameters
        ----------
        z_map : torch.Tensor
            Latent tensor of shape (B, latent_dim, H, W)

        Returns
        -------
        torch.Tensor
            Reconstructed image tensor of shape (B, out_channels, H*4, W*4)
        """
        h = self.from_latent(z_map)
        h = self.dec_block1(h)  # -> (B, 64, H*2, W*2)
        h = self.dec_block2(h)  # -> (B, proj_channels, H*4, W*4)
        x_hat = self.out_conv(h)  # -> (B, out_channels, H*4, W*4)
        return x_hat


class AutoEncoder(nn.Module):
    """
    Complete autoencoder with spatial latent representation for astronomical images.

    This implements an architecture that preserves spatial structure
    in the latent representation.

    The model uses a multi-stage encoder that downsamples the input by a factor of 4,
    resulting in a spatial latent representation. For a 96x96 input, the latent
    representation is 24x24 with a specified number of channels (latent_dim).

    Parameters
    ----------
    in_channels : int
        Number of input image channels (e.g., 12 for S-PLUS bands)
    latent_dim : int, default=2
        Number of latent channels per spatial position
    proj_channels : int, default=64
        Number of channels in intermediate projection layers

    Attributes
    ----------
    encoder : Encoder
        Encoder network
    decoder : Decoder
        Decoder network
    latent_H : int
        Height of latent spatial representation
    latent_W : int
        Width of latent spatial representation

    Examples
    --------
    >>> # Create autoencoder for 12-band images with 2-channel latent
    >>> model = AutoEncoder(in_channels=12, latent_dim=2)
    >>> x = torch.randn(4, 12, 96, 96)
    >>> x_hat, z_map = model(x)
    >>> x_hat.shape
    torch.Size([4, 12, 96, 96])
    >>> z_map.shape
    torch.Size([4, 2, 24, 24])

    >>> # Separate encode and decode steps
    >>> z = model.encode(x)
    >>> x_reconstructed = model.decode(z)

    Notes
    -----
    The architecture uses:
    - ResBlocks for feature learning at each scale
    - Strided convolutions for downsampling in encoder
    - Transposed convolutions for upsampling in decoder
    - 1x1 convolutions for dimensionality changes
    - ReLU activations throughout
    """

    def __init__(
        self,
        in_channels: int,
        latent_dim: int = 2,
        proj_channels: int = 64,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.proj_channels = proj_channels

        self.encoder = Encoder(in_channels, latent_dim, proj_channels)
        self.decoder = Decoder(latent_dim, in_channels, proj_channels)

        # Store latent spatial dimensions from encoder
        self.latent_H = self.encoder.latent_H
        self.latent_W = self.encoder.latent_W

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input image to latent representation.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (B, C, H, W)

        Returns
        -------
        torch.Tensor
            Latent representation of shape (B, latent_dim, H/4, W/4)

        Examples
        --------
        >>> model = AutoEncoder(in_channels=12, latent_dim=2)
        >>> x = torch.randn(4, 12, 96, 96)
        >>> z = model.encode(x)
        >>> z.shape
        torch.Size([4, 2, 24, 24])
        """
        return self.encoder(x)

    def decode(self, z_map: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to image.

        Parameters
        ----------
        z_map : torch.Tensor
            Latent representation of shape (B, latent_dim, H, W)

        Returns
        -------
        torch.Tensor
            Reconstructed image of shape (B, C, H*4, W*4)

        Examples
        --------
        >>> model = AutoEncoder(in_channels=12, latent_dim=2)
        >>> z = torch.randn(4, 2, 24, 24)
        >>> x_hat = model.decode(z)
        >>> x_hat.shape
        torch.Size([4, 12, 96, 96])
        """
        return self.decoder(z_map)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through autoencoder.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (B, C, H, W)

        Returns
        -------
        x_hat : torch.Tensor
            Reconstructed image of shape (B, C, H, W)
        z_map : torch.Tensor
            Latent representation of shape (B, latent_dim, H/4, W/4)

        Examples
        --------
        >>> model = AutoEncoder(in_channels=12, latent_dim=2)
        >>> x = torch.randn(4, 12, 96, 96)
        >>> x_hat, z_map = model(x)
        >>> x_hat.shape
        torch.Size([4, 12, 96, 96])
        >>> z_map.shape
        torch.Size([4, 2, 24, 24])
        """
        z_map = self.encode(x)
        x_hat = self.decode(z_map)
        return x_hat, z_map
