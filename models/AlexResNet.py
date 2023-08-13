import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import os.path
import numpy as np


def compute_conv_output_size(Lin, kernel_size, stride=1, padding=0, dilation=1):
    return int(np.floor((Lin + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1))


def gen_mask(x):
    return x


class batchEnsemble_AlexNet(nn.Module):
    def __init__(self, currentModel, head, BN, WMask_Sperturb, WMask_Rperturb):
        super(batchEnsemble_AlexNet, self).__init__()
        assert isinstance(currentModel, AlexNet), "The model class should be LeNet!"

        self.features = nn.Sequential(
            batchEnsemble_Conv2d(
                currentModel.features[0], WMask_Sperturb["0"], WMask_Rperturb["0"]
            ),
            BN["1"],
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            batchEnsemble_Conv2d(
                currentModel.features[5], WMask_Sperturb["5"], WMask_Rperturb["5"]
            ),
            BN["6"],
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            batchEnsemble_Conv2d(
                currentModel.features[10], WMask_Sperturb["10"], WMask_Rperturb["10"]
            ),
            BN["11"],
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            View(currentModel.features[16].in_features),
            batchEnsemble_Linear(
                currentModel.features[16], WMask_Sperturb["16"], WMask_Rperturb["16"]
            ),
            # nn.BatchNorm2d(2048, track_running_stats=False),
            BN["17"],
            nn.ReLU(),
            nn.Dropout(0.5),
            batchEnsemble_Linear(
                currentModel.features[20], WMask_Sperturb["20"], WMask_Rperturb["20"]
            ),
            BN["21"],
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.last = head

    def forward(self, x):
        x = self.features(x)
        x = self.last(x)
        return x


#########################
class rankPerturb4_Conv2d(nn.Module):
    def __init__(
        self, layerW, layerB, mask_S, mask_R, param_S, param_R, kernel_size, padding, stride=1
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
        x1 = F.conv2d(x1, weight=self.layerW, padding=self.padding, stride=self.stride) * gen_mask(
            self.mask_R
        )  # channelwise filter
        x1 = x1 + self.layerB.view(1, self.out_channels, 1, 1)
        x2 = F.conv2d(x, weight=self.kernel1)
        x2 = (
            F.avg_pool2d(x2, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
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


# 3
class View(nn.Module):
    def __init__(self, n_feat):
        super(View, self).__init__()
        self.n_feat = n_feat

    def forward(self, x):
        return x.view(-1, self.n_feat)


class AlexNet(nn.Module):
    def __init__(self, out_dim, img_sz=32):
        super(AlexNet, self).__init__()
        s = compute_conv_output_size(img_sz, 4)
        s = s // 2
        s = compute_conv_output_size(s, 3)
        s = s // 2
        s = compute_conv_output_size(s, 2)
        s = s // 2
        smid = s

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 4, bias=True),
            nn.BatchNorm2d(64, track_running_stats=False),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, bias=True),
            nn.BatchNorm2d(128, track_running_stats=False),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 2, bias=True),
            nn.BatchNorm2d(256, track_running_stats=False),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool2d(2),
            View(256 * smid * smid),
            nn.Linear(256 * smid * smid, 2048, bias=True),
            nn.BatchNorm1d(2048, track_running_stats=False),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048, bias=True),
            nn.BatchNorm1d(2048, track_running_stats=False),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.last = nn.Linear(2048, out_dim, bias=True)

    def forward(self, x):
        x = self.features(x)
        x = self.last(x)
        return x


# 3


class rankPerturb4_AlexNet(nn.Module):
    def __init__(
        self,
        currentModel,
        Bias,
        head,
        BN,
        WMask_Sperturb,
        WMask_Rperturb,
        WBias_Sperturb,
        WBias_Rperturb,
    ):
        super(rankPerturb4_AlexNet, self).__init__()
        assert isinstance(currentModel, AlexNet), "The model class should be AlexNet!"
        # print(currentModel)

        self.features = nn.Sequential(
            rankPerturb4_Conv2d(
                currentModel.features[0].weight,
                Bias["0"],
                WMask_Sperturb["0"],
                WMask_Rperturb["0"],
                WBias_Sperturb["0"],
                WBias_Rperturb["0"],
                kernel_size=[4, 4],
                padding=0,
                stride=1,
            ),
            # nn.ReLU(inplace=True),
            # nn.BatchNorm2d(64, track_running_stats=False),
            BN["1"],
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(64, 192, kernel_size=5, padding=2),
            rankPerturb4_Conv2d(
                currentModel.features[5].weight,
                Bias["5"],
                WMask_Sperturb["5"],
                WMask_Rperturb["5"],
                WBias_Sperturb["5"],
                WBias_Rperturb["5"],
                kernel_size=[3, 3],
                padding=0,
                stride=1,
            ),
            # nn.BatchNorm2d(128, track_running_stats=False),
            BN["6"],
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            rankPerturb4_Conv2d(
                currentModel.features[10].weight,
                Bias["10"],
                WMask_Sperturb["10"],
                WMask_Rperturb["10"],
                WBias_Sperturb["10"],
                WBias_Rperturb["10"],
                kernel_size=[2, 2],
                padding=0,
                stride=1,
            ),
            BN["11"],
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            View(currentModel.features[16].in_features),
            rankPerturb4_Linear(
                currentModel.features[16].weight,
                Bias["16"],
                WMask_Sperturb["16"],
                WMask_Rperturb["16"],
                WBias_Sperturb["16"],
                WBias_Rperturb["16"],
            ),
            BN["17"],
            nn.ReLU(),
            nn.Dropout(0.5),
            rankPerturb4_Linear(
                currentModel.features[20].weight,
                Bias["20"],
                WMask_Sperturb["20"],
                WMask_Rperturb["20"],
                WBias_Sperturb["20"],
                WBias_Rperturb["20"],
            ),
            # nn.BatchNorm2d(2048, track_running_stats=False),
            BN["21"],
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.last = head

    def forward(self, x):
        x = self.features(x)
        x = self.last(x)
        return x


class apdPerturb_AlexNet(nn.Module):
    def __init__(self, param_share, Bias, head, BN, WMask_Rperturb, WBias_Rperturb, img_sz=32):
        super(apdPerturb_AlexNet, self).__init__()
        # assert isinstance(currentModel,AlexNet), 'The model class should be AlexNet!'
        # print(currentModel)
        s = compute_conv_output_size(img_sz, 4)
        s = s // 2

        s = compute_conv_output_size(s, 3)
        s = s // 2

        s = compute_conv_output_size(s, 2)
        s = s // 2
        smid = s
        n_convfeat = 256 * smid * smid

        self.features = nn.Sequential(
            apdPerturb_Conv2d(
                param_share["0"], Bias["0"], WMask_Rperturb["0"], WBias_Rperturb["0"], padding=0
            ),
            BN["1"],
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            apdPerturb_Conv2d(
                param_share["5"], Bias["5"], WMask_Rperturb["5"], WBias_Rperturb["5"], padding=0
            ),
            BN["6"],
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            apdPerturb_Conv2d(
                param_share["10"], Bias["10"], WMask_Rperturb["10"], WBias_Rperturb["10"], padding=0
            ),
            BN["11"],
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            View(n_convfeat),
            apdPerturb_Linear(
                param_share["16"], Bias["16"], WMask_Rperturb["16"], WBias_Rperturb["16"]
            ),
            BN["17"],
            nn.ReLU(),
            nn.Dropout(0.5),
            apdPerturb_Linear(
                param_share["20"], Bias["20"], WMask_Rperturb["20"], WBias_Rperturb["20"]
            ),
            BN["21"],
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.last = head

    def forward(self, x):
        x = self.features(x)
        x = self.last(x)
        return x


class batchNorm_AlexNet(nn.Module):
    def __init__(self, currentModel, head, BN):
        super(batchNorm_AlexNet, self).__init__()
        assert isinstance(currentModel, AlexNet), "The model class should be AlexNet!"

        self.features = nn.Sequential(
            currentModel.features[0],
            BN["1"],
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            currentModel.features[5],
            BN["6"],
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            currentModel.features[10],
            BN["11"],
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            View(currentModel.features[16].in_features),
            currentModel.features[16],
            BN["17"],
            nn.ReLU(),
            nn.Dropout(0.5),
            currentModel.features[20],
            BN["21"],
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.last = head

    def forward(self, x):
        x = self.features(x)
        x = self.last(x)
        return x


###################
# Original
class changeHead(nn.Module):
    def __init__(self, model, newhead):
        super(changeHead, self).__init__()
        self.features = model.features
        self.last = newhead

    def forward(self, x):
        x = self.features(x)
        x = self.last(x)
        return x


class apdPerturb_Conv2d(nn.Module):
    def __init__(self, layerW, layerB, mask_R, param_R, padding):
        super(apdPerturb_Conv2d, self).__init__()
        self.layerW = layerW
        self.layerB = layerB
        self.padding = padding
        self.out_channels = len(layerB)
        self.mask_R = mask_R.view(self.out_channels, 1, 1, 1)
        self.kernel2 = param_R

    def forward(self, x):
        perturb_weight = self.layerW * gen_mask(self.mask_R) + self.kernel2
        x = F.conv2d(x, weight=perturb_weight, padding=self.padding)
        x = x + self.layerB.view(1, self.out_channels, 1, 1)
        return x


class apdPerturb_Linear(nn.Module):
    def __init__(self, layerW, layerB, mask_R, param_R):
        super(apdPerturb_Linear, self).__init__()
        self.layerW = layerW
        self.layerB = layerB
        self.out_features = len(layerB)
        self.mask_R = mask_R.view(self.out_features, 1)
        self.kernel2 = param_R

    def forward(self, x):
        perturb_w = self.layerW * gen_mask(self.mask_R) + self.kernel2
        x = F.linear(x, weight=perturb_w) + self.layerB.view(1, self.out_features)
        return x


class batchEnsemble_Conv2d(nn.Module):
    def __init__(self, layer, mask_S, mask_R):
        super(batchEnsemble_Conv2d, self).__init__()
        self.layer = layer
        self.mask_S = mask_S.view(1, layer.in_channels, 1, 1)
        self.mask_R = mask_R.view(1, layer.out_channels, 1, 1)

    def forward(self, x):
        # channelwise filter
        x = x * self.mask_S
        x = F.conv2d(x, weight=self.layer.weight, padding=self.layer.padding)
        x = x * self.mask_R + self.layer.bias.view(1, self.layer.out_channels, 1, 1)
        return x


class batchEnsemble_Linear(nn.Module):
    def __init__(self, layer, mask_S, mask_R):
        super(batchEnsemble_Linear, self).__init__()
        self.layer = layer
        self.mask_S = mask_S.view(1, layer.in_features)
        self.mask_R = mask_R.view(1, layer.out_features)

    def forward(self, x):
        x = x * self.mask_S
        x = F.linear(x, weight=self.layer.weight)
        x = x * self.mask_R + self.layer.bias.view(1, self.layer.out_features)
        return x


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
