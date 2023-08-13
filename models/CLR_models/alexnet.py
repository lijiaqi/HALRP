from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic import View, compute_conv_output_size
from .clr import CLRBlock

__all__ = ["AlexNet", "CLR_AlexNet", "batchNorm_CLR_AlexNet"]


class AlexNet(nn.Module):
    def __init__(self, num_cls, img_sz=32, track_bn_stats=True):
        super(AlexNet, self).__init__()
        # self.act=OrderedDict()
        # self.map =[]
        # self.ksize=[]
        # self.in_channel =[]
        # self.map.append(32)
        # self.conv1 = nn.Conv2d(3, 64, 4, bias=False)
        # self.bn1 = nn.BatchNorm2d(64, track_running_stats=False)
        s = compute_conv_output_size(img_sz, 4)
        s = s // 2
        # self.ksize.append(4)
        # self.in_channel.append(3)
        # self.map.append(s)
        # self.conv2 = nn.Conv2d(64, 128, 3, bias=False)
        # self.bn2 = nn.BatchNorm2d(128, track_running_stats=False)
        s = compute_conv_output_size(s, 3)
        s = s // 2
        # self.ksize.append(3)
        # self.in_channel.append(64)
        # self.map.append(s)
        # self.conv3 = nn.Conv2d(128, 256, 2, bias=False)
        # self.bn3 = nn.BatchNorm2d(256, track_running_stats=False)
        s = compute_conv_output_size(s, 2)
        s = s // 2
        smid = s
        # self.ksize.append(2)
        # self.in_channel.append(128)
        # self.map.append(256*self.smid*self.smid)
        # self.maxpool=torch.nn.MaxPool2d(2)
        # self.relu=torch.nn.ReLU()
        # self.drop1=torch.nn.Dropout(0.2)
        # self.drop2=torch.nn.Dropout(0.5)

        # self.fc1 = nn.Linear(256*self.smid*self.smid,2048, bias=False)
        # self.bn4 = nn.BatchNorm1d(2048, track_running_stats=False)
        # self.fc2 = nn.Linear(2048,2048, bias=False)
        # self.bn5 = nn.BatchNorm1d(2048, track_running_stats=False)
        # self.map.extend([2048])

        # self.taskcla = taskcla
        # self.fc3=torch.nn.ModuleList()
        # for t,n in self.taskcla:
        #    self.fc3.append(torch.nn.Linear(2048,n,bias=False))

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 4, bias=True),
            nn.BatchNorm2d(64, track_running_stats=track_bn_stats),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, bias=True),
            nn.BatchNorm2d(128, track_running_stats=track_bn_stats),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 2, bias=True),
            nn.BatchNorm2d(256, track_running_stats=track_bn_stats),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool2d(2),
            View(256 * smid * smid),
            nn.Linear(256 * smid * smid, 2048, bias=True),
            nn.BatchNorm1d(2048, track_running_stats=track_bn_stats),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048, bias=True),
            nn.BatchNorm1d(2048, track_running_stats=track_bn_stats),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.backbone_dim = 2048
        self.classifier = nn.Linear(self.backbone_dim, num_cls)

    def forward(self, x):
        feat = self.features(x)
        logits = self.classifier(feat)
        return logits


class CLR_AlexNet(nn.Module):

    def __init__(self, currentModel, newClassifier):
        super(CLR_AlexNet, self).__init__()
        assert isinstance(currentModel, AlexNet), "The model class should be AlexNet!"
        self.clr_blocks_list = []
        new_layers = []
        for idx, layer in enumerate(currentModel.features):
            if idx in [0, 5, 10]:
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


class batchNorm_CLR_AlexNet(nn.Module):

    def __init__(self, currentModel, BN, newClassifier):
        super(batchNorm_CLR_AlexNet, self).__init__()
        assert isinstance(currentModel, AlexNet), "The model class should be AlexNet!"
        self.clr_blocks_list = []
        new_layers = []
        for idx, layer in enumerate(currentModel.features):
            if idx in [0, 5, 10]:
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
