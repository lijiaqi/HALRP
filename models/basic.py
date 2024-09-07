# -*- coding: utf-8 -*-
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


def gen_mask(x):
    return x


class View(nn.Module):
    def __init__(self, n_feat):
        super(View, self).__init__()
        self.n_feat = n_feat

    def forward(self, x):
        return x.view(-1, self.n_feat)


class changeHead(nn.Module):
    def __init__(self, model, newhead):
        super(changeHead, self).__init__()
        self.features = model.features
        self.last = newhead

    def forward(self, x):
        x = self.features(x)
        x = self.last(x)
        return x


class rankPerturb4_Conv2d(nn.Module):

    def __init__(
        self,
        layerW,
        layerB,
        mask_S,
        mask_R,
        param_S,
        param_R,
        kernel_size,
        padding,
        stride=1,
    ):
        super(rankPerturb4_Conv2d, self).__init__()
        self.layerW = layerW
        self.layerB = layerB
        self.padding = padding
        self.stride = stride

        self.in_channels = layerW.shape[1]
        self.out_channels = layerW.shape[0]
        self.mask_S = mask_S.view(1, self.in_channels, 1, 1)
        self.mask_R = mask_R.view(1, self.out_channels, 1, 1)
        self.kernel1 = param_S.view(param_S.shape[0], self.in_channels, 1, 1)
        self.kernel2 = param_R.view(self.out_channels, param_S.shape[0], 1, 1)
        self.kernel_size = kernel_size

    def forward(self, x):
        x1 = x * gen_mask(self.mask_S)
        x1 = F.conv2d(
            x1, weight=self.layerW, padding=self.padding, stride=self.stride
        ) * gen_mask(
            self.mask_R
        )  # channelwise filter
        x1 = x1 + self.layerB.view(1, self.out_channels, 1, 1)
        x2 = F.conv2d(x, weight=self.kernel1)
        x2 = (
            F.avg_pool2d(
                x2, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding
            )
            * self.kernel_size[0]
            * self.kernel_size[1]
        )
        x2 = F.conv2d(x2, weight=self.kernel2)
        x = x1 + x2
        return x

    def extra_repr(self):
        return "Rank:{}".format(self.kernel1.shape[0])


class rankPerturb4_Linear(nn.Module):
    def __init__(self, layerW, layerB, mask_S, mask_R, param_S, param_R):
        super(rankPerturb4_Linear, self).__init__()
        self.layerW = layerW
        self.layerB = layerB
        self.in_features = layerW.shape[1]
        self.out_features = layerW.shape[0]
        self.mask_S = mask_S.view(1, self.in_features)
        self.mask_R = mask_R.view(1, self.out_features)
        self.kernel1 = param_S.view(param_S.shape[0], self.in_features)
        self.kernel2 = param_R.view(self.out_features, param_S.shape[0])

    def forward(self, x):
        x1 = x * gen_mask(self.mask_S)
        x1 = F.linear(x1, weight=self.layerW) * gen_mask(self.mask_R)
        x1 = x1 + self.layerB.view(1, self.out_features)
        x2 = F.linear(x, weight=self.kernel1)
        x2 = F.linear(x2, weight=self.kernel2)
        x = x1 + x2
        return x

    def extra_repr(self):
        return "Rank:{}".format(self.kernel1.shape[0])
