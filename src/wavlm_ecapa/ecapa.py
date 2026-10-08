# Adapted from Microsoft UniSpeech / lawlict ECAPA-TDNN.
# See LICENSE and NOTICE.md in this package.
import torch
from torch import nn
from torch.nn import functional as F


class Res2Conv1dReluBn(nn.Module):
    """
    in_channels == out_channels == channels
    """

    def __init__(
        self, channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, scale=4
    ):
        super().__init__()
        assert channels % scale == 0, "{} % {} != 0".format(channels, scale)
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1
        self.convs = []
        self.bns = []
        for i in range(self.nums):
            self.convs.append(
                nn.Conv1d(self.width, self.width, kernel_size, stride, padding, dilation, bias=bias)
            )
            self.bns.append(nn.BatchNorm1d(self.width))
        self.convs = nn.ModuleList(self.convs)
        self.bns = nn.ModuleList(self.bns)

    def forward(self, x):
        out = []
        spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.bns[i](F.relu(sp))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        out = torch.cat(out, dim=1)
        return out


class Conv1dReluBn(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias
        )
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))


class SE_Connect(nn.Module):
    def __init__(self, channels, se_bottleneck_dim=128):
        super().__init__()
        self.linear1 = nn.Linear(channels, se_bottleneck_dim)
        self.linear2 = nn.Linear(se_bottleneck_dim, channels)

    def forward(self, x):
        out = x.mean(dim=2)
        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        out = x * out.unsqueeze(2)
        return out


class SE_Res2Block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        scale,
        se_bottleneck_dim,
    ):
        super().__init__()
        self.Conv1dReluBn1 = Conv1dReluBn(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.Res2Conv1dReluBn = Res2Conv1dReluBn(
            out_channels, kernel_size, stride, padding, dilation, scale=scale
        )
        self.Conv1dReluBn2 = Conv1dReluBn(
            out_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.SE_Connect = SE_Connect(out_channels, se_bottleneck_dim)
        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1
            )

    def forward(self, x):
        residual = x
        if self.shortcut:
            residual = self.shortcut(x)
        x = self.Conv1dReluBn1(x)
        x = self.Res2Conv1dReluBn(x)
        x = self.Conv1dReluBn2(x)
        x = self.SE_Connect(x)
        return x + residual


class AttentiveStatsPool(nn.Module):
    def __init__(self, in_dim, attention_channels=128, global_context_att=False):
        super().__init__()
        self.global_context_att = global_context_att
        if global_context_att:
            self.linear1 = nn.Conv1d(in_dim * 3, attention_channels, kernel_size=1)
        else:
            self.linear1 = nn.Conv1d(in_dim, attention_channels, kernel_size=1)
        self.linear2 = nn.Conv1d(attention_channels, in_dim, kernel_size=1)

    def forward(self, x):
        if self.global_context_att:
            context_mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
            context_std = torch.sqrt(torch.var(x, dim=-1, keepdim=True) + 1e-10).expand_as(x)
            x_in = torch.cat((x, context_mean, context_std), dim=1)
        else:
            x_in = x
        alpha = torch.tanh(self.linear1(x_in))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        residuals = torch.sum(alpha * x**2, dim=2) - mean**2
        std = torch.sqrt(residuals.clamp(min=1e-09))
        return torch.cat([mean, std], dim=1)
