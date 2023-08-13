import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from .basic import ProjectionMLP, View, compute_conv_output_size


class AlexNet(nn.Module):
    """backbone + projection head"""

    def __init__(self, input_size=32, track_bn_stats=False):
        super(AlexNet, self).__init__()
        # self.act=OrderedDict()
        # self.map =[]
        # self.ksize=[]
        # self.in_channel =[]
        # self.map.append(32)
        # self.conv1 = nn.Conv2d(3, 64, 4, bias=False)
        # self.bn1 = nn.BatchNorm2d(64, track_running_stats=False)
        s = compute_conv_output_size(input_size, 4)
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
        self.encoder = nn.Sequential(
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
        self.last_hid = 2048

    def forward(self, x):
        feat = self.encoder(x)
        return feat


class SupConAlexNet(nn.Module):
    """backbone + projection head"""

    def __init__(
        self,
        head="mlp",
        input_size=32,
        track_bn_stats=False,
        feat_dim=128,
        hidden_dim=256,
        proj_bn=False,
        num_layers=2,
    ):
        super(SupConAlexNet, self).__init__()
        self.encoder = AlexNet(input_size=input_size, track_bn_stats=track_bn_stats)

        if head == "linear":
            self.head = nn.Linear(self.encoder.last_hid, feat_dim)
        elif head == "mlp":
            self.head = ProjectionMLP(
                self.encoder.last_hid, hidden_dim, feat_dim, proj_bn, num_layers
            )
        else:
            raise NotImplementedError("head not supported: {}".format(head))

    def return_hidden(self, x, layer=-1):
        return self.encoder(x)

    def forward_classifier(self, x, task=None):
        feat = self.head(x)
        return feat

    def forward(self, x, task=None):
        feat = self.head(self.encoder(x))
        return feat
