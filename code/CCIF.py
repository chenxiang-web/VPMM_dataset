import torch
import torch.nn as nn


class CCIF(nn.Module):
    """
    CCIF: Cross-modal Channel Interaction Fusion.

    This module enhances channel interaction through grouped gating and
    channel permutation/rearrangement, enabling adaptive cross-channel
    fusion on 2D feature maps.

    Args:
        group (int): Number of channel groups. Default: 4.
    """
    def __init__(self, group: int = 4) -> None:
        super().__init__()
        self.group = int(group)
        self.gating: nn.Module | None = None
        self._c: int | None = None

    def _build_if_needed(self, c: int) -> None:
        """Build grouped gating branch dynamically based on channel size."""
        if self.gating is not None and self._c == c:
            return
        if c % self.group != 0:
            raise ValueError(f"CCIF: channel dimension {c} is not divisible by group number {self.group}")
        self.gating = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                c, c,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=self.group,
                bias=True,
            ),
            nn.Sigmoid(),
        )
        self._c = c

    def channel_interaction(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shuffle channels across groups to promote cross-group interaction.
        """
        b, c, h, w = x.shape
        g = self.group
        if c % g != 0:
            raise ValueError(f"CCIF.channel_interaction: channel dimension {c} is not divisible by group number {g}")
        gc = c // g
        x = x.reshape(b, gc, g, h, w).permute(0, 2, 1, 3, 4).reshape(b, c, h, w)
        return x

    def channel_fusion(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rearrange channels back after grouped gating for channel fusion.
        """
        b, c, h, w = x.shape
        g = self.group
        if c % g != 0:
            raise ValueError(f"CCIF.channel_fusion: channel dimension {c} is not divisible by group number {g}")
        gc = c // g
        x = x.reshape(b, g, gc, h, w).permute(0, 2, 1, 3, 4).reshape(b, c, h, w)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of CCIF.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W].

        Returns:
            torch.Tensor: Channel-enhanced fused feature map.
        """
        if not isinstance(x, torch.Tensor) or x.dim() != 4:
            raise TypeError("CCIF expects input with shape [B, C, H, W]")

        _, c, _, _ = x.shape
        self._build_if_needed(c)

        residual = x
        x = self.channel_interaction(x)
        g = self.gating(x)
        g = self.channel_fusion(g)

        return residual * g