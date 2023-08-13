# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic import View, rankPerturb4_Conv2d, rankPerturb4_Linear

__all__ = ["LeNet", "LargeLeNet", "rankPerturb4_LeNet"]


class LeNet(nn.Module):
    def __init__(self, out_dim=10, in_channel=3, img_sz=32, track_bn_stats=True):
        super(LeNet, self).__init__()
        feat_map_sz = img_sz // 4
        n_convfeat = 50 * feat_map_sz * feat_map_sz

        self.features = nn.Sequential(
            nn.Conv2d(in_channel, 20, 5, padding=2),
            nn.BatchNorm2d(20, track_running_stats=track_bn_stats),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(20, 50, 5, padding=2),
            nn.BatchNorm2d(50, track_running_stats=track_bn_stats),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            View(n_convfeat),
            nn.Linear(n_convfeat, 800),
            nn.BatchNorm1d(800, track_running_stats=track_bn_stats),
            nn.ReLU(inplace=True),
            nn.Linear(800, 500),
            nn.BatchNorm1d(500, track_running_stats=track_bn_stats),
            nn.ReLU(inplace=True),
        )

        self.last = nn.Linear(500, out_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.last(x)
        return x


class rankPerturb4_LeNet(nn.Module):

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
        super(rankPerturb4_LeNet, self).__init__()
        assert isinstance(
            currentModel, (LeNet, LargeLeNet)
        ), "The model class should be LeNet or LargeLeNet!"

        self.features = nn.Sequential(
            rankPerturb4_Conv2d(
                currentModel.features[0].weight,
                Bias["0"],
                WMask_Sperturb["0"],
                WMask_Rperturb["0"],
                WBias_Sperturb["0"],
                WBias_Rperturb["0"],
                kernel_size=[5, 5],
                padding=2,
            ),
            BN["1"],
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            rankPerturb4_Conv2d(
                currentModel.features[4].weight,
                Bias["4"],
                WMask_Sperturb["4"],
                WMask_Rperturb["4"],
                WBias_Sperturb["4"],
                WBias_Rperturb["4"],
                kernel_size=[5, 5],
                padding=2,
            ),
            BN["5"],
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            View(currentModel.features[9].in_features),
            rankPerturb4_Linear(
                currentModel.features[9].weight,
                Bias["9"],
                WMask_Sperturb["9"],
                WMask_Rperturb["9"],
                WBias_Sperturb["9"],
                WBias_Rperturb["9"],
            ),
            BN["10"],
            nn.ReLU(inplace=True),
            rankPerturb4_Linear(
                currentModel.features[12].weight,
                Bias["12"],
                WMask_Sperturb["12"],
                WMask_Rperturb["12"],
                WBias_Sperturb["12"],
                WBias_Rperturb["12"],
            ),
            BN["13"],
            nn.ReLU(inplace=True),
        )

        self.last = head

    def forward(self, x):
        x = self.features(x)
        x = self.last(x)
        return x


### for omniglot-rotation dataset
class LargeLeNet(nn.Module):
    def __init__(self, out_dim=10, in_channel=1, img_sz=32, track_bn_stats=True):
        super(LargeLeNet, self).__init__()
        feat_map_sz = img_sz // 4
        n_convfeat = 50 * feat_map_sz * feat_map_sz

        self.features = nn.Sequential(
            nn.Conv2d(in_channel, 64, 5, padding=2),
            nn.BatchNorm2d(20, track_running_stats=track_bn_stats),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 5, padding=2),
            nn.BatchNorm2d(128, track_running_stats=track_bn_stats),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            View(n_convfeat),
            nn.Linear(n_convfeat, 2500),
            nn.BatchNorm1d(2500, track_running_stats=track_bn_stats),
            nn.ReLU(inplace=True),
            nn.Linear(2500, 1500),
            nn.BatchNorm1d(1500, track_running_stats=track_bn_stats),
            nn.ReLU(inplace=True),
        )

        self.last = nn.Linear(1500, out_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.last(x)
        return x
