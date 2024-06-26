import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic import ProjectionMLP


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False, track_bn_stats=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=track_bn_stats)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=track_bn_stats)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    self.expansion * planes, track_running_stats=track_bn_stats
                ),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False, track_bn_stats=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=track_bn_stats)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=track_bn_stats)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(
            self.expansion * planes, track_running_stats=track_bn_stats
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    self.expansion * planes, track_running_stats=track_bn_stats
                ),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):

    def __init__(
        self,
        block,
        num_blocks,
        nf,
        input_size,
        in_channel=3,
        zero_init_residual=False,
        track_bn_stats=False,
    ):
        super(ResNet, self).__init__()

        self.in_planes = nf
        self.input_size = input_size

        # hardcoded for now
        self.last_hid = nf * 8 * block.expansion
        # self.last_hid = last_hid * (input_size[-1] // 2 // 2 // 2 // 4) ** 2

        self.conv1 = nn.Conv2d(
            in_channel, nf, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(nf, track_running_stats=track_bn_stats)
        self.layer1 = self._make_layer(
            block, 1 * nf, num_blocks[0], stride=1, track_bn_stats=track_bn_stats
        )
        self.layer2 = self._make_layer(
            block, 2 * nf, num_blocks[1], stride=2, track_bn_stats=track_bn_stats
        )
        self.layer3 = self._make_layer(
            block, 4 * nf, num_blocks[2], stride=2, track_bn_stats=track_bn_stats
        )
        self.layer4 = self._make_layer(
            block, 8 * nf, num_blocks[3], stride=2, track_bn_stats=track_bn_stats
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride, track_bn_stats):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(
                block(self.in_planes, planes, stride, track_bn_stats=track_bn_stats)
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def return_hidden(self, x, layer):
        if layer < 1 or layer > 4:
            layer = 4
        layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        out = F.relu(self.bn1(self.conv1(x)))
        for lyr in layers[:layer]:
            out = lyr(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


class SupConResNet18(nn.Module):
    """backbone + projection head"""

    def __init__(
        self,
        head="mlp",
        nf=32,
        input_size=32,
        track_bn_stats=False,
        feat_dim=128,
        hidden_dim=256,
        proj_bn=False,
        num_layers=2,
    ):
        super(SupConResNet18, self).__init__()
        self.encoder = resnet18(nf=nf, input_size=input_size, track_bn_stats=track_bn_stats)

        if head == "linear":
            self.head = nn.Linear(self.encoder.last_hid, feat_dim)
        elif head == "mlp":
            self.head = ProjectionMLP(
                self.encoder.last_hid, hidden_dim, feat_dim, proj_bn, num_layers
            )
        else:
            raise NotImplementedError("head not supported: {}".format(head))

    def return_hidden(self, x, layer=-1):
        return self.encoder.return_hidden(x, layer)

    def forward_classifier(self, x, task=None):
        feat = self.head(x)
        return feat

    def forward(self, x, task=None):
        feat = self.head(self.encoder(x))
        # feat = F.normalize(feat, dim=1)
        return feat
