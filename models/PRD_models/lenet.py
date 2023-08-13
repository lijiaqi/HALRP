import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic import ProjectionMLP, View


class LeNet(nn.Module):
    """backbone + projection head"""

    def __init__(self, input_size=32, in_channel=3, track_bn_stats=False):
        super(LeNet, self).__init__()
        feat_map_sz = input_size // 4
        n_convfeat = 50 * feat_map_sz * feat_map_sz
        self.encoder = nn.Sequential(
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
        self.last_hid = 500

    def forward(self, x):
        feat = self.encoder(x)
        return feat


class SupConLeNet(nn.Module):
    """backbone + projection head"""

    def __init__(
        self,
        head="mlp",
        input_size=32,
        in_channel=3,
        track_bn_stats=False,
        feat_dim=128,
        hidden_dim=256,
        proj_bn=False,
        num_layers=2,
    ):
        super(SupConLeNet, self).__init__()
        self.encoder = LeNet(
            input_size=input_size, in_channel=in_channel, track_bn_stats=track_bn_stats
        )

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


class WideLeNet(nn.Module):
    """backbone + projection head"""

    def __init__(self, input_size=32, in_channel=1, track_bn_stats=False):
        super(WideLeNet, self).__init__()
        feat_map_sz = input_size // 4
        n_convfeat = 128 * feat_map_sz * feat_map_sz
        self.encoder = nn.Sequential(
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
        self.last_hid = 1500

    def forward(self, x):
        feat = self.encoder(x)
        return feat

class SupConWideLeNet(nn.Module):
    """backbone + projection head"""

    def __init__(
        self,
        head="mlp",
        input_size=32,
        in_channel=1,
        track_bn_stats=False,
        feat_dim=128,
        hidden_dim=256,
        proj_bn=False,
        num_layers=2,
    ):
        super(SupConWideLeNet, self).__init__()
        self.encoder = WideLeNet(
            input_size=input_size, in_channel=in_channel, track_bn_stats=track_bn_stats
        )

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
