import math
import torch
from torch import nn


@torch.jit.script
def snake(x: torch.Tensor, alpha):
    x_shape = x.shape
    min_add = 1e-9

    x = x.reshape(x_shape[0], x_shape[1], -1)
    x = x + (alpha + min_add).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(x_shape)

    return x


class SnakeBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, input_x: torch.Tensor):
        output = snake(input_x, self.alpha)

        return output


# conv blocks with weight norm
class Conv1D_wn(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = nn.Conv1d(*args, **kwargs)

    def forward(self, x):
        output = nn.utils.weight_norm(self.conv(x))

        return output


# resnet_like layers/units
class ResLayer(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, dilation: int):
        super().__init__()

        self.dilation = dilation
        padding = (dilation * (7 - 1)) // 2

        self.snake_conv_block = nn.Sequential(
            SnakeBlock(ch_out),
            Conv1D_wn(
                ch_in,
                ch_out,
                kernel_size=7,
                stride=1,
                padding=padding,
                dilation=dilation,
            ),
            SnakeBlock(ch_out),
            Conv1D_wn(ch_out, ch_out, kernel_size=1, stride=1),
        )

    def forward(self, x: torch.Tensor):
        res_copy = x

        x = self.conv1(x)
        x = self.snake_activate(x)
        x = self.conv2(x)

        return x + res_copy


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.downsample_layer = nn.Sequential(
            ResLayer(in_ch, in_ch, dilation=1),
            ResLayer(in_ch, in_ch, dilation=3),
            ResLayer(in_ch, in_ch, dilation=9),
            SnakeBlock(in_ch),
            Conv1D_wn(
                in_ch,
                out_ch,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        x = self.downsample_layer(x)

        return x
