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


# resnet_like layers/units
class ResnetLayer(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.snake_conv_block = nn.Sequential(
            SnakeBlock(ch_out),
            nn.Conv1d(ch_in, ch_out, kernel_size=7, stride=1, dilation=2),
            SnakeBlock(ch_out),
            nn.Conv1d(ch_out, ch_out * 3, kernel_size=1, stride=1),
        )

    def forward(self, x: torch.Tensor):
        res_copy = x

        x = self.conv1(x)
        x = self.snake_activate(x)
        x = self.conv2(x)

        return x + res_copy
