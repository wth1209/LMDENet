import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['SoftEdgeConcat']


class SoftEdgeConcat(nn.Module):
    """
    输入 (B, C, H, W) → 输出 (B, 2C, H, W)
    对每个通道独立提取水平/垂直边缘，然后用可训练的 α、β 做柔和加权，再与原输入在通道维拼接。
    - α、β 为逐通道参数，通过 softplus 保证非负，并归一化使 α+β=1，训练更稳定。
    - Sobel 与可选高斯核为固定 buffer（不参与训练）。
    """
    def __init__(self,
                 c1: int,                   # 输入通道数
                 alpha_init: float = 0.5,   # α 初值（逐通道参数会初始化为该值）
                 beta_init: float = 0.5,    # β 初值
                 sobel_ksize: int = 3,      # 目前实现 3x3
                 blur_ksize: int = 0,       # 可选高斯平滑核，0 代表不启用；需为奇数
                 blur_sigma: float = 0.0,   # 高斯σ，>0 生效
                 eps: float = 1e-6):
        super().__init__()
        assert sobel_ksize == 3, "当前实现仅支持 3x3 Sobel"
        if blur_ksize:
            assert blur_ksize % 2 == 1, "blur_ksize 必须为奇数(3/5/7/...)"

        self.c1 = c1
        self.c2 = 2 * c1
        self.eps = eps

        # ------------------ Sobel（固定，不训练） ------------------
        gx = torch.tensor([[-1., 0., 1.],
                           [-2., 0., 2.],
                           [-1., 0., 1.]], dtype=torch.float32)
        gy = gx.t()
        sobel_x = torch.zeros((c1, 1, 3, 3), dtype=torch.float32)
        sobel_y = torch.zeros((c1, 1, 3, 3), dtype=torch.float32)
        for i in range(c1):
            sobel_x[i, 0] = gx
            sobel_y[i, 0] = gy
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)

        # ------------------ 可选高斯（固定，不训练，可分离） ------------------
        if blur_ksize and blur_sigma > 0:
            half = blur_ksize // 2
            x = torch.arange(-half, half + 1, dtype=torch.float32)
            g1d = torch.exp(-(x ** 2) / (2 * (blur_sigma ** 2)))
            g1d = g1d / (g1d.sum() + 1e-8)
            gk_x = torch.zeros((c1, 1, 1, blur_ksize), dtype=torch.float32)
            gk_y = torch.zeros((c1, 1, blur_ksize, 1), dtype=torch.float32)
            for i in range(c1):
                gk_x[i, 0, 0, :] = g1d
                gk_y[i, 0, :, 0] = g1d
            self.register_buffer("gk_x", gk_x, persistent=False)
            self.register_buffer("gk_y", gk_y, persistent=False)
            self.blur_ksize = blur_ksize
        else:
            self.gk_x = None
            self.gk_y = None

        # ------------------ 可训练 α、β（逐通道） ------------------
        # 用 raw 参数 + softplus → 保证非负，再做归一化到和为1
        self.alpha_raw = nn.Parameter(torch.full((c1, 1, 1), float(alpha_init)))
        self.beta_raw  = nn.Parameter(torch.full((c1, 1, 1), float(beta_init)))
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) → out: (B, 2C, H, W)  = concat([x, edge_fused], dim=1)
        """
        B, C, H, W = x.shape
        assert C == self.c1, f"通道不匹配：输入 {C}，模块期望 {self.c1}"

        # Sobel（depthwise）
        gx = F.conv2d(x, self.sobel_x, stride=1, padding=1, groups=C)
        gy = F.conv2d(x, self.sobel_y, stride=1, padding=1, groups=C)

        # 逐通道 α、β（正且归一化）
        a = self.softplus(self.alpha_raw)   # (C,1,1) ≥ 0
        b = self.softplus(self.beta_raw)    # (C,1,1) ≥ 0
        s = (a + b + self.eps)
        a = a / s
        b = b / s

        # 柔和加权融合：edge = α*|gx| + β*|gy|（逐通道）
        edge = a.unsqueeze(0) * gx.abs() + b.unsqueeze(0) * gy.abs()

        # 可选高斯平滑（depthwise，可分离）
        if self.gk_x is not None and self.gk_y is not None:
            edge = F.conv2d(edge, self.gk_x, padding=(0, self.blur_ksize // 2), groups=C)
            edge = F.conv2d(edge, self.gk_y, padding=(self.blur_ksize // 2, 0), groups=C)

        # 与原始输入级联 → 输出通道 2C
        out = torch.cat([x, edge], dim=1)
        return out
