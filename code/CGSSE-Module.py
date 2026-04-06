# CGSSE Module (Cross-modal Global-Structural Spectral Enhancement Module)
import torch
import torch.nn as nn

class EDFFN(nn.Module):
    """
    Enhanced Dynamic Feed-Forward Network.
    This module performs dynamic feed-forward transformation with patch-wise
    spectral processing based on FFT.

    Args:
        dim (int): Number of input channels.
        ffn_expansion_factor (float): Expansion ratio of the hidden dimension.
            Default: 2.
        bias (bool): Whether to use bias in convolution layers. Default: False.
    """
    def __init__(self, dim, ffn_expansion_factor=2, bias=False):
        super(EDFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 8
        self.dim = dim

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias
        )

        self.fft = nn.Parameter(
            torch.ones((dim, 1, 1, self.patch_size, self.patch_size // 2 + 1))
        )

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x_dtype = x.dtype

        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)

        b, c, h, w = x.shape

        h_n = (self.patch_size - h % self.patch_size) % self.patch_size
        w_n = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, w_n, 0, h_n), mode='reflect')

        x_patch = rearrange(
            x,
            'b c (h patch1) (w patch2) -> b c h w patch1 patch2',
            patch1=self.patch_size,
            patch2=self.patch_size
        )

        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(
            x_patch_fft,
            s=(self.patch_size, self.patch_size)
        )

        x = rearrange(
            x_patch,
            'b c h w patch1 patch2 -> b c (h patch1) (w patch2)',
            patch1=self.patch_size,
            patch2=self.patch_size
        )

        x = x[:, :, :h, :w]
        return x.to(x_dtype)


class LayerNorm2d(nn.LayerNorm):
    """
    2D Layer Normalization for feature maps.

    This layer applies LayerNorm over the channel dimension of a 2D feature map
    with shape (B, C, H, W).
    """
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class MonaOp(nn.Module):
    """
    Mona operation with multi-scale depthwise convolutions.
    This operator uses three depthwise convolutions with different receptive
    fields (3x3, 5x5, 7x7), followed by projection and residual fusion.

    Args:
        in_features (int): Number of input channels.
    """
    def __init__(self, in_features):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_features, in_features, kernel_size=3, padding=1, groups=in_features
        )
        self.conv2 = nn.Conv2d(
            in_features, in_features, kernel_size=5, padding=2, groups=in_features
        )
        self.conv3 = nn.Conv2d(
            in_features, in_features, kernel_size=7, padding=3, groups=in_features
        )
        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1)

    def forward(self, x):
        identity = x

        conv1_x = self.conv1(x)
        conv2_x = self.conv2(x)
        conv3_x = self.conv3(x)

        x = (conv1_x + conv2_x + conv3_x) / 3.0 + identity
        identity = x
        x = self.projector(x)

        return identity + x


class Mona(nn.Module):
    """
    Modular Attention Normalization module.
    This module enhances feature normalization by combining LayerNorm,
    learnable scaling, dimensionality reduction, multi-scale convolutional
    adaptation, nonlinearity, and residual projection.

    Args:
        in_dim (int): Number of input channels.
    """
    def __init__(self, in_dim):
        super().__init__()

        self.project1 = nn.Conv2d(in_dim, 64, 1)
        self.nonlinear = F.gelu
        self.project2 = nn.Conv2d(64, in_dim, 1)
        self.dropout = nn.Dropout(p=0.1)
        self.adapter_conv = MonaOp(64)

        self.norm = LayerNorm2d(in_dim)
        self.gamma = nn.Parameter(torch.ones(in_dim, 1, 1) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim, 1, 1))

    def forward(self, x, hw_shapes=None):
        identity = x

        x = self.norm(x) * self.gamma + x * self.gammax
        x = self.project1(x)
        x = self.adapter_conv(x)
        x = self.nonlinear(x)
        x = self.dropout(x)
        x = self.project2(x)

        return identity + x


class DynamicTanh(nn.Module):
    """
    Dynamic Tanh normalization module.

    Source:
        CVPR 2025

    This module applies adaptive tanh activation with a learnable alpha
    parameter, followed by affine transformation.

    Args:
        normalized_shape (int): Number of normalized channels.
        channels_last (bool): Whether the input uses channels-last format.
        alpha_init_value (float): Initial value of alpha. Default: 0.5.
    """
    def __init__(self, normalized_shape, channels_last, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x

    def extra_repr(self):
        return (
            f"normalized_shape={self.normalized_shape}, "
            f"alpha_init_value={self.alpha_init_value}, "
            f"channels_last={self.channels_last}"
        )


class CGSSEBlock(PSABlock):
    """
    CGSSE Block: Cross-modal Global Structure and Spectral Enhancement block.

    This block integrates four enhancement mechanisms into a unified PSA-style
    architecture:
        1. Token Statistics Self-Attention (TSSA)
        2. DynamicTanh normalization
        3. Mona normalization/adaptation
        4. Enhanced Dynamic Feed-Forward Network (EDFFN)

    The design aims to jointly improve global dependency modeling, structural
    representation, and spectral enhancement capability.

    Args:
        c (int): Number of input channels.
        attn_ratio (float): Attention ratio. Default: 0.5.
        num_heads (int): Number of attention heads. Default: 4.
        shortcut (bool): Whether to use shortcut connection. Default: True.
    """
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__(c, attn_ratio, num_heads, shortcut)

        self.ffn = EDFFN(dim=c, ffn_expansion_factor=2.66, bias=False)

        self.dyt1 = DynamicTanh(normalized_shape=c, channels_last=False)
        self.dyt2 = DynamicTanh(normalized_shape=c, channels_last=False)

        self.mona1 = Mona(c)
        self.mona2 = Mona(c)

        self.attn = AttentionTSSA(c, num_heads=num_heads)

    def forward(self, x):
        b, c, h, w = x.size()

        attn_out = self.attn(
            self.dyt1(x).flatten(2).permute(0, 2, 1)
        ).permute(0, 2, 1).view(-1, c, h, w).contiguous()

        x = x + attn_out if self.add else attn_out
        x = self.mona1(x)

        ffn_out = self.ffn(self.dyt2(x))
        x = x + ffn_out if self.add else ffn_out

        x = self.mona2(x)
        return x


class CGSSEModule(C2PSA):
    """
    CGSSE Module: Cross-modal Global Structure and Spectral Enhancement Module.

    This module is a C2PSA-based variant that stacks multiple CGSSE blocks
    to jointly model token-level global interactions, adaptive normalization,
    structural feature refinement, and spectral-domain enhancement.

    Integrated components:
        - TSSA for global token statistics modeling
        - DynamicTanh for adaptive activation normalization
        - Mona for structural normalization and local adaptation
        - EDFFN for patch-wise spectral enhancement

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int): Number of stacked CGSSE blocks. Default: 1.
        e (float): Expansion ratio. Default: 0.5.

    Example:
        >>> cgsse = CGSSEModule(c1=256, c2=256, n=3, e=0.5)
        >>> x = torch.randn(1, 256, 64, 64)
        >>> y = cgsse(x)
    """
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        self.m = nn.Sequential(
            *(CGSSEBlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))
        )