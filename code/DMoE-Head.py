# DMoE-Head (Directional Mixture-of-Experts Head)
class Scale(nn.Module):
    """A learnable scale parameter for DMoE-Head.

    This layer scales the input by a learnable factor.

    Args:
        scale (float): Initial value of scale factor. Default: 1.0.
    """

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


def autopad(k, p=None, d=1):
    """Automatic padding calculation."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv_GN(nn.Module):
    """Convolution + GroupNorm + Activation block used in DMoE-Head."""
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, gn_groups=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        groups = min(gn_groups, c2)
        self.gn = nn.GroupNorm(groups, c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))


class DirectionalMoEConv(nn.Module):
    """
    Directional Mixture-of-Experts Convolution for DMoE-Head.

    This module decomposes features into four directional experts and
    adaptively fuses them with a lightweight gating mechanism.

    Design:
        - Four directional branches act as four experts
        - Direction-aware asymmetric convolutions model structural patterns
        - Softmax gating assigns expert importance dynamically
        - A fusion convolution integrates expert outputs
    """

    def __init__(self, c1, c2, k=3, s=1, gate_reduction=8, gn_groups=16):
        super().__init__()
        self.use_directional_moe = (c2 % 4 == 0)
        self.k = k

        if not self.use_directional_moe:
            # Fallback for channel numbers not divisible by 4
            self.fallback = nn.Sequential(
                Conv_GN(c1, c2, 3, s=1, g=c2, gn_groups=gn_groups),
                Conv_GN(c2, c2, 1, s=1, gn_groups=gn_groups),
            )
            return

        # Direction-aware padding for the four experts
        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = nn.ModuleList([nn.ZeroPad2d(padding=p[i]) for i in range(4)])

        c_branch = c2 // 4
        self.horizontal_expert = Conv_GN(c1, c_branch, (1, k), s=s, p=0, gn_groups=gn_groups)
        self.vertical_expert = Conv_GN(c1, c_branch, (k, 1), s=s, p=0, gn_groups=gn_groups)

        hidden = max(1, c_branch // gate_reduction)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_branch, hidden, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, 1, bias=True),
        )

        self.fuse = Conv_GN(c2, c2, 2, s=1, p=0, gn_groups=gn_groups)

    def forward(self, x):
        if not self.use_directional_moe:
            return self.fallback(x)

        e0 = self.horizontal_expert(self.pad[0](x))
        e1 = self.horizontal_expert(self.pad[1](x))
        e2 = self.vertical_expert(self.pad[2](x))
        e3 = self.vertical_expert(self.pad[3](x))
        experts = [e0, e1, e2, e3]

        scores = torch.cat([self.gate(e) for e in experts], dim=1)  # (B, 4, 1, 1)
        weights = F.softmax(scores, dim=1)  # (B, 4, 1, 1)

        experts = [experts[i] * weights[:, i:i + 1, :, :] for i in range(4)]
        y = torch.cat(experts, dim=1)
        return self.fuse(y)


class DMoESharedConv(nn.Module):
    """
    Shared convolution block for DMoE-Head.

    This block applies directional mixture-of-experts convolution followed by
    pointwise projection, with residual learning for more stable optimization.
    """

    def __init__(self, c, k=3, gn_groups=16):
        super().__init__()
        self.directional_moe = DirectionalMoEConv(
            c, c, k=k, s=1, gate_reduction=8, gn_groups=gn_groups
        )
        self.pw = Conv_GN(c, c, 1, s=1, gn_groups=gn_groups)

    def forward(self, x):
        return x + self.pw(self.directional_moe(x))


class DMoEHead(nn.Module):
    """
    DMoE-Head: Directional Mixture-of-Experts Head.

    This detection head introduces directional mixture-of-experts modeling
    into a lightweight shared-convolution detection framework.

    Key features:
        - Direction-aware expert decomposition
        - Lightweight gating-based expert fusion
        - Shared convolution across detection levels
        - GroupNorm for stable training
        - Learnable scale for per-level regression adjustment

    Args:
        nc (int): Number of classes.
        hidc (int): Hidden channels for intermediate layers.
        ch (tuple): Input channels from backbone/neck for each detection level.
    """

    dynamic = False
    export = False
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, hidc=256, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)

        # Per-level feature adaptation
        self.conv = nn.ModuleList(nn.Sequential(Conv_GN(x, hidc, 3)) for x in ch)

        # Shared DMoE block across detection levels
        self.share_conv = DMoESharedConv(hidc, k=3, gn_groups=16)

        # Output heads
        self.cv2 = nn.Conv2d(hidc, 4 * self.reg_max, 1)  # bbox regression
        self.cv3 = nn.Conv2d(hidc, self.nc, 1)           # classification

        # Learnable scale for each detection level
        self.scale = nn.ModuleList(Scale(1.0) for _ in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Forward pass of DMoE-Head."""
        for i in range(self.nl):
            x[i] = self.conv[i](x[i])
            x[i] = self.share_conv(x[i])
            x[i] = torch.cat((self.scale[i](self.cv2(x[i])), self.cv3(x[i])), 1)

        if self.training:
            return x

        shape = x[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)

        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (
                x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5)
            )
            self.shape = shape

        if self.export and self.format in ("saved_model", "pb", "tflite", "edgetpu", "tfjs"):
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        dbox = self.decode_bboxes(box)

        if self.export and self.format in ("tflite", "edgetpu"):
            img_h = shape[2]
            img_w = shape[3]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * img_size)
            dbox = dist2bbox(
                self.dfl(box) * norm,
                self.anchors.unsqueeze(0) * norm[:, :2],
                xywh=True,
                dim=1
            )

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize biases for DMoE-Head."""
        m = self
        m.cv2.bias.data[:] = 1.0
        m.cv3.bias.data[: m.nc] = math.log(5 / m.nc / (640 / 16) ** 2)

    def decode_bboxes(self, bboxes):
        """Decode predicted bounding boxes."""
        return dist2bbox(
            self.dfl(bboxes),
            self.anchors.unsqueeze(0),
            xywh=True,
            dim=1
        ) * self.strides