import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def compute_conv_output_size(Lin, kernel_size, stride=1, padding=0, dilation=1):
    return int(
        np.floor(
            (Lin + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1
        )
    )


class View(nn.Module):
    def __init__(self, n_feat):
        super(View, self).__init__()
        self.n_feat = n_feat

    def forward(self, x):
        return x.view(-1, self.n_feat)
