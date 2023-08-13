import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic import View
from .clr import CLRBlock

__all__ = ["LeNet", "WideLeNet", "CLR_LeNet", "batchNorm_CLR_LeNet"]


class LeNet(nn.Module):
    def __init__(self, num_cls, in_channel=3, img_sz=32, track_bn_stats=True):
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
        self.backbone_dim = 500
        self.classifier = nn.Linear(self.backbone_dim, num_cls)

    def forward(self, x):
        feat = self.features(x)
        logits = self.classifier(feat)
        return logits


class WideLeNet(nn.Module):
    def __init__(self, num_cls, in_channel=3, img_sz=32, track_bn_stats=True):
        super(WideLeNet, self).__init__()
        feat_map_sz = img_sz // 4
        n_convfeat = 128 * feat_map_sz * feat_map_sz
        self.features = nn.Sequential(
            nn.Conv2d(in_channel, 64, 5, padding=2),
            nn.BatchNorm2d(64, track_running_stats=track_bn_stats),
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
        self.backbone_dim = 1500
        self.classifier = nn.Linear(self.backbone_dim, num_cls)

    def forward(self, x):
        feat = self.features(x)
        logits = self.classifier(feat)
        return logits


class CLR_LeNet(nn.Module):

    def __init__(self, currentModel, newClassifier):
        super(CLR_LeNet, self).__init__()
        assert isinstance(
            currentModel, (LeNet, WideLeNet)
        ), "The model class should be 'LeNet' or 'WideLeNet'!"
        self.clr_blocks_list = []
        new_layers = []
        for idx, layer in enumerate(currentModel.features):
            if idx in [0, 4]:
                clr_block = CLRBlock(layer, kernel_size=3)
                new_layers.append(clr_block)
                self.clr_blocks_list.append(clr_block)
            else:
                new_layers.append(layer)
        self.features = nn.Sequential(*new_layers)

        self.backbone_dim = currentModel.backbone_dim
        self.classifier = newClassifier

    def forward(self, x):
        feat = self.features(x)
        logits = self.classifier(feat)
        return logits

    def get_clr_blocks(self):
        return self.clr_blocks_list


class batchNorm_CLR_LeNet(nn.Module):

    def __init__(self, currentModel, BN, newClassifier):
        super(batchNorm_CLR_LeNet, self).__init__()
        assert isinstance(
            currentModel, (LeNet, WideLeNet)
        ), "The model class should be 'LeNet' or 'WideLeNet'!"
        self.clr_blocks_list = []
        new_layers = []
        for idx, layer in enumerate(currentModel.features):
            if idx in [0, 4]:
                clr_block = CLRBlock(layer, kernel_size=3)
                new_layers.append(clr_block)
                self.clr_blocks_list.append(clr_block)
            elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
                new_layers.append(BN[str(idx)])
            else:
                new_layers.append(layer)
        self.features = nn.Sequential(*new_layers)

        self.backbone_dim = currentModel.backbone_dim
        self.classifier = newClassifier

    def forward(self, x):
        feat = self.features(x)
        logits = self.classifier(feat)
        return logits

    def get_clr_blocks(self):
        return self.clr_blocks_list
