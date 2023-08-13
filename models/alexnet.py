import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic import compute_conv_output_size
from .basic import View, rankPerturb4_Conv2d, rankPerturb4_Linear

__all__ = ["AlexNet", "rankPerturb4_AlexNet"]

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
