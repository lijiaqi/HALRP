import torch
import torch.nn as nn
import torch
import torch.nn as nn

import copy


class CLRBlock(nn.Module):
    def __init__(self, block, kernel_size=3):
        super(CLRBlock, self).__init__()
        self.kernel_size = kernel_size
        # self.original_conv = copy.deepcopy(block)
        self.original_conv = block

        for param in self.original_conv.parameters():
            param.requires_grad = False
        self.out_channels = self.original_conv.out_channels
        # initialize with Identity layer for first task
        self.clr_conv = nn.Conv2d(
            self.out_channels,
            self.out_channels,
            self.kernel_size,
            1,
            self.kernel_size // 2,
            groups=self.out_channels,
            bias=False,
        )
        nn.init.dirac_(self.clr_conv.weight, groups=self.out_channels)

    def forward(self, x):
        output = self.clr_conv(self.original_conv(x))
        return output


def set_BN(model, requires_grad=False):
    for name, layer in model.named_children():
        # if isinstance(layer, nn.BatchNorm2d): # original, only set 2D BN
        if isinstance(layer, (nn.BatchNorm2d, nn.BatchNorm1d)):
            model._modules[name].train()
            for param in model._modules[name].parameters():
                param.requires_grad = requires_grad
        set_BN(layer, requires_grad)
    return model


def add_parameters(params, model, layer_type):
    for name, layer in model.named_children():
        if isinstance(layer, layer_type):
            params += list(layer.parameters())
        params = add_parameters(params, layer, layer_type)
    return params


# apply to a simple net
def Conv2CLR(model):
    for name, layer in model.named_children():
        if isinstance(layer, torch.nn.modules.conv.Conv2d):
            # no linear transformation for 1*1 convolutional
            if layer.kernel_size != (1, 1):
                model._modules[name] = CLRBlock(layer)
        else:
            Conv2CLR(layer)
    return model
