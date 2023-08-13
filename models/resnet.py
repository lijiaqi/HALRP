from collections import OrderedDict

from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic import rankPerturb4_Conv2d, rankPerturb4_Linear

__all__ = ["changeResNet18Head", "ResNet_flat_18", "rankPerturb4_Res18"]


class changeResNet18Head(nn.Module):
    def __init__(self, model, newhead):
        super(changeResNet18Head, self).__init__()
        self.features = model.features

        self.block0 = model.block0
        # layer 1
        self.block1 = model.block1
        self.shortcut1 = model.shortcut1
        self.block2 = model.block2
        self.shortcut2 = model.shortcut2
        # layer 2
        self.block3 = model.block3
        self.shortcut3 = model.shortcut3
        self.block4 = model.block4
        self.shortcut4 = model.shortcut4
        # layer 3
        self.block5 = model.block5
        self.shortcut5 = model.shortcut5
        self.block6 = model.block6
        self.shortcut6 = model.shortcut6
        # layer 4
        self.block7 = model.block7
        self.shortcut7 = model.shortcut7
        self.block8 = model.block8
        self.shortcut8 = model.shortcut8
        # head
        self.last = newhead

    def forward(self, x):
        out = self.block0(x)
        out = F.relu(self.block1(out) + self.shortcut1(out))
        out = F.relu(self.block2(out) + self.shortcut2(out))
        out = F.relu(self.block3(out) + self.shortcut3(out))
        out = F.relu(self.block4(out) + self.shortcut4(out))
        out = F.relu(self.block5(out) + self.shortcut5(out))
        out = F.relu(self.block6(out) + self.shortcut6(out))
        out = F.relu(self.block7(out) + self.shortcut7(out))
        out = F.relu(self.block8(out) + self.shortcut8(out))
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.last(out)
        return out


########################
# original Resnet18
########################
# Define ResNet18 model
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True
    )


def conv7x7(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=7, stride=stride, padding=1, bias=True
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, track_bn_stats=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=track_bn_stats)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=track_bn_stats)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=True,
                ),
                nn.BatchNorm2d(
                    self.expansion * planes, track_running_stats=track_bn_stats
                ),
            )
        self.act = OrderedDict()
        self.count = 0

    def forward(self, x):
        self.count = self.count % 2
        self.act["conv_{}".format(self.count)] = x
        self.count += 1
        out = F.relu(self.bn1(self.conv1(x)))
        self.count = self.count % 2
        self.act["conv_{}".format(self.count)] = out
        self.count += 1
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(
        self,
        block,
        num_blocks,
        out_dim=10,
        nf=20,
        n_task=5,
        multi_head=False,
        track_bn_stats=False,
        img_sz=32,
    ):
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.img_sz = img_sz
        self.track_bn_stats = track_bn_stats
        self.conv1 = conv3x3(3, nf * 1, 1)
        self.bn1 = nn.BatchNorm2d(nf * 1, track_running_stats=track_bn_stats)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        ss = img_sz // (2**4)  # 3 conv(stride=2) + 1 avgpool(stride=2)

        # self.taskcla = taskcla
        self.multi_head = multi_head
        self.n_task = n_task
        self.linear = torch.nn.ModuleList()

        if multi_head:
            for t in range(n_task):
                self.linear.append(
                    nn.Linear(nf * 8 * block.expansion * ss * ss, out_dim, bias=True)
                )
        else:
            self.linear.append(
                nn.Linear(nf * 8 * block.expansion * ss * ss, out_dim, bias=True)
            )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(self.in_planes, planes, stride, track_bn_stats=self.track_bn_stats)
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        bsz = x.size(0)
        self.act["conv_in"] = x.view(bsz, 3, self.img_sz, self.img_sz)
        out = F.relu(self.bn1(self.conv1(x.view(bsz, 3, self.img_sz, self.img_sz))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 2)

        out = out.view(out.size(0), -1)
        y = []

        if self.multi_head:
            for t in range(self.n_task):
                y.append(self.linear[t](out))
        else:
            y = self.linear(out)

        return y


def construct_flat_Res18(nf=32, out_dim=10, n_task=5, img_sz=32, track_bn_stats=False):
    def get_children(model: torch.nn.Module):
        children = list(model.children())
        flatt_children = []
        if children == []:
            return [model]
        else:
            for child in children:
                flatt_children.extend(get_children(child))
        return flatt_children

    temp_model = ResNet(
        block=BasicBlock,
        num_blocks=[2, 2, 2, 2],
        out_dim=out_dim,
        nf=nf,
        n_task=n_task,
        img_sz=img_sz,
        track_bn_stats=track_bn_stats,
    )
    new_model = nn.Sequential()
    for i, j in enumerate(get_children(temp_model)):
        new_model.add_module(str(i), j)
    return new_model


class ResNet_flat_18(nn.Module):

    def __init__(self, nf=32, out_dim=10, n_task=5, img_sz=32, track_bn_stats=False):
        super(ResNet_flat_18, self).__init__()
        self.img_sz = img_sz
        self.features = nn.Sequential()

        self.block0 = nn.Sequential()
        # layer 1
        self.block1 = nn.Sequential()
        self.shortcut1 = nn.Sequential()
        self.block2 = nn.Sequential()
        self.shortcut2 = nn.Sequential()
        # layer 2
        self.block3 = nn.Sequential()
        self.shortcut3 = nn.Sequential()
        self.block4 = nn.Sequential()
        self.shortcut4 = nn.Sequential()
        # layer 3
        self.block5 = nn.Sequential()
        self.shortcut5 = nn.Sequential()
        self.block6 = nn.Sequential()
        self.shortcut6 = nn.Sequential()
        # layer 4
        self.block7 = nn.Sequential()
        self.shortcut7 = nn.Sequential()
        self.block8 = nn.Sequential()
        self.shortcut8 = nn.Sequential()
        # head
        self.last = nn.Sequential()

        resmodel = construct_flat_Res18(
            nf=nf,
            out_dim=out_dim,
            n_task=n_task,
            img_sz=img_sz,
            track_bn_stats=track_bn_stats,
        )

        # for idx in relu_index:
        # |    self.features.add_module(str(m), nn.ReLU())
        for i, j in enumerate(resmodel.children()):
            # print(i)
            ### block0: conv1, bn1
            if i < 2:
                self.block0.add_module(str(i), j)
                self.features.add_module(str(i), j)
                # add relu
                if i == 1:
                    self.block0.add_module(str(i + 1), nn.ReLU())
                    self.features.add_module(str(i + 1), nn.ReLU())

            ### layer1-block1
            elif i >= 2 and i <= 3:
                self.block1.add_module(str(i + 1), j)
                self.features.add_module(str(i + 1), j)
                # add relu
                if i == 3:
                    self.block1.add_module(str(i + 2), nn.ReLU())
                    self.features.add_module(str(i + 2), nn.ReLU())
            elif i >= 4 and i <= 5:
                self.block1.add_module(str(i + 2), j)
                self.features.add_module(str(i + 2), j)
            ### layer1-shortcut1
            elif i == 6:
                self.shortcut1 = j
                self.features.add_module(str(i + 2), j)

            ### layer1-block2
            elif i >= 7 and i <= 8:
                self.block2.add_module(str(i + 2), j)
                self.features.add_module(str(i + 2), j)
                # add relu
                if i == 8:
                    self.block2.add_module(str(i + 3), nn.ReLU())
                    self.features.add_module(str(i + 3), nn.ReLU())
            elif i >= 9 and i <= 10:
                self.block2.add_module(str(i + 3), j)
                self.features.add_module(str(i + 3), j)
            ### layer1-shortcut2
            elif i == 11:
                self.shortcut2 = j
                self.features.add_module(str(i + 3), j)

            ### layer2-block3
            elif i >= 12 and i <= 13:
                self.block3.add_module(str(i + 3), j)
                self.features.add_module(str(i + 3), j)
                # add relu
                if i == 13:
                    self.block3.add_module(str(i + 4), nn.ReLU())
                    self.features.add_module(str(i + 4), nn.ReLU())
            elif i >= 14 and i <= 15:
                self.block3.add_module(str(i + 4), j)
                self.features.add_module(str(i + 4), j)
            ### layer2-shortcut3
            elif i >= 16 and i <= 17:
                self.shortcut3.add_module(str(i + 4), j)
                self.features.add_module(str(i + 4), j)
            ### layer2-block4
            elif i >= 18 and i <= 19:
                self.block4.add_module(str(i + 4), j)
                self.features.add_module(str(i + 4), j)
                # add relu
                if i == 19:
                    self.block4.add_module(str(i + 5), nn.ReLU())
                    self.features.add_module(str(i + 5), nn.ReLU())
            elif i >= 20 and i <= 21:
                self.block4.add_module(str(i + 5), j)
                self.features.add_module(str(i + 5), j)
            ### layer2-shortcut4
            elif i == 22:
                self.shortcut4 = j
                self.features.add_module(str(i + 5), j)

            ### layer-3, block-5
            elif i >= 23 and i <= 24:
                self.block5.add_module(str(i + 5), j)
                self.features.add_module(str(i + 5), j)
                # add relu
                if i == 24:
                    self.block5.add_module(str(i + 6), nn.ReLU())
                    self.features.add_module(str(i + 6), nn.ReLU())
            elif i >= 25 and i <= 26:
                self.block5.add_module(str(i + 6), j)
                self.features.add_module(str(i + 6), j)
            ### layer-3, shortcut-5
            elif i >= 27 and i <= 28:
                self.shortcut5.add_module(str(i + 6), j)
                self.features.add_module(str(i + 6), j)
            ### layer-3, block-6
            elif i >= 29 and i <= 30:
                self.block6.add_module(str(i + 6), j)
                self.features.add_module(str(i + 6), j)
                # add relu
                if i == 30:
                    self.block6.add_module(str(i + 7), nn.ReLU())
                    self.features.add_module(str(i + 7), nn.ReLU())
            elif i >= 31 and i <= 32:
                self.block6.add_module(str(i + 7), j)
                self.features.add_module(str(i + 7), j)
            ### layer-3, shortcut-6
            elif i == 33:
                self.shortcut6 = j
                self.features.add_module(str(i + 7), j)

            ### layer-4, block-7
            elif i >= 34 and i <= 35:
                self.block7.add_module(str(i + 7), j)
                self.features.add_module(str(i + 7), j)
                # add relu
                if i == 35:
                    self.block7.add_module(str(i + 8), nn.ReLU())
                    self.features.add_module(str(i + 8), nn.ReLU())
            elif i >= 36 and i <= 37:
                self.block7.add_module(str(i + 8), j)
                self.features.add_module(str(i + 8), j)
            ### layer-4, shortcut-7
            elif i >= 38 and i <= 39:
                self.shortcut7.add_module(str(i + 8), j)
                self.features.add_module(str(i + 8), j)
            ### layer-4, block-8
            elif i >= 40 and i <= 41:
                self.block8.add_module(str(i + 8), j)
                self.features.add_module(str(i + 8), j)
                # add relu
                if i == 41:
                    self.block8.add_module(str(i + 9), nn.ReLU())
                    self.features.add_module(str(i + 9), nn.ReLU())
            elif i >= 42 and i <= 43:
                self.block8.add_module(str(i + 9), j)
                self.features.add_module(str(i + 9), j)
            ### layer-4, shortcut-8
            elif i == 44:
                self.shortcut8 = j
                self.features.add_module(str(i + 9), j)
            ### head
            elif i == 45:
                self.last = j

    def forward(self, x):
        out = self.block0(x)
        out = F.relu(self.block1(out) + self.shortcut1(out))
        out = F.relu(self.block2(out) + self.shortcut2(out))
        out = F.relu(self.block3(out) + self.shortcut3(out))
        out = F.relu(self.block4(out) + self.shortcut4(out))
        out = F.relu(self.block5(out) + self.shortcut5(out))
        out = F.relu(self.block6(out) + self.shortcut6(out))
        out = F.relu(self.block7(out) + self.shortcut7(out))
        out = F.relu(self.block8(out) + self.shortcut8(out))
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.last(out)
        return out


class rankPerturb4_Res18(nn.Module):

    def __init__(
        self,
        currentModel,  # model on task-0
        Bias,
        head,
        BN,
        WMask_Sperturb,
        WMask_Rperturb,
        WBias_Sperturb,
        WBias_Rperturb,
    ):
        super(rankPerturb4_Res18, self).__init__()
        assert isinstance(
            currentModel, ResNet_flat_18
        ), "The model class should be ResNet18!"
        # print(currentModel)

        self.features = nn.Sequential()
        for i, j in enumerate(currentModel.features):
            current_layer = currentModel.features[i]
            if isinstance(j, nn.Conv2d):
                self.features.add_module(
                    str(i),
                    rankPerturb4_Conv2d(
                        current_layer.weight,
                        Bias[str(i)],
                        WMask_Sperturb[str(i)],
                        WMask_Rperturb[str(i)],
                        WBias_Sperturb[str(i)],
                        WBias_Rperturb[str(i)],
                        kernel_size=[
                            current_layer.kernel_size[0],
                            current_layer.kernel_size[1],
                        ],
                        padding=current_layer.padding[0],
                        stride=current_layer.stride[0],
                    ),
                )
            elif isinstance(j, nn.Linear):
                self.features.add_module(
                    str(i),
                    rankPerturb4_Linear(
                        current_layer.weight,
                        Bias[str(i)],
                        WMask_Sperturb[str(i)],
                        WMask_Rperturb[str(i)],
                        WBias_Sperturb[str(i)],
                        WBias_Rperturb[str(i)],
                    ),
                )
            elif isinstance(
                j, (torch.nn.modules.BatchNorm2d, torch.nn.modules.BatchNorm1d)
            ):
                self.features.add_module(str(i), BN[str(i)])
            else:  # Relu()/Sequential()/...
                self.features.add_module(str(i), deepcopy(j))

        self.block0 = nn.Sequential()
        # layer 1
        self.block1 = nn.Sequential()
        self.shortcut1 = nn.Sequential()
        self.block2 = nn.Sequential()
        self.shortcut2 = nn.Sequential()
        # layer 2
        self.block3 = nn.Sequential()
        self.shortcut3 = nn.Sequential()
        self.block4 = nn.Sequential()
        self.shortcut4 = nn.Sequential()
        # layer 3
        self.block5 = nn.Sequential()
        self.shortcut5 = nn.Sequential()
        self.block6 = nn.Sequential()
        self.shortcut6 = nn.Sequential()
        # layer 4
        self.block7 = nn.Sequential()
        self.shortcut7 = nn.Sequential()
        self.block8 = nn.Sequential()
        self.shortcut8 = nn.Sequential()
        # head
        self.last = head

        for i, j in enumerate(self.features.named_children()):
            num = i
            layer = j[1]
            if num <= 2:
                self.block0.add_module(str(num), layer)
            # layer 1
            elif num > 2 and num <= 7:
                self.block1.add_module(str(num), layer)
            elif num == 8:
                self.shortcut1 = layer
            elif num > 8 and num <= 13:
                self.block2.add_module(str(num), layer)
            elif num == 14:
                self.shortcut2 = layer
            # layer 2
            elif num > 14 and num <= 19:
                self.block3.add_module(str(num), layer)
            elif num > 19 and num <= 21:
                self.shortcut3.add_module(str(num), layer)
            elif num > 21 and num <= 26:
                self.block4.add_module(str(num), layer)
            elif num == 27:
                self.shortcut4 = layer
            # layer 3
            elif num > 27 and num <= 32:
                self.block5.add_module(str(num), layer)
            elif num > 32 and num <= 34:
                self.shortcut5.add_module(str(num), layer)
            elif num > 34 and num <= 39:
                self.block6.add_module(str(num), layer)
            elif num == 40:
                self.shortcut6 = layer
            # layer 4
            elif num > 40 and num <= 45:
                self.block7.add_module(str(num), layer)
            elif num > 45 and num <= 47:
                self.shortcut7.add_module(str(num), layer)
            elif num > 47 and num <= 52:
                self.block8.add_module(str(num), layer)
            elif num == 53:
                self.shortcut8 = layer

    def forward(self, x):
        out = self.block0(x)
        out = F.relu(self.block1(out) + self.shortcut1(out))
        out = F.relu(self.block2(out) + self.shortcut2(out))
        out = F.relu(self.block3(out) + self.shortcut3(out))
        out = F.relu(self.block4(out) + self.shortcut4(out))
        out = F.relu(self.block5(out) + self.shortcut5(out))
        out = F.relu(self.block6(out) + self.shortcut6(out))
        out = F.relu(self.block7(out) + self.shortcut7(out))
        out = F.relu(self.block8(out) + self.shortcut8(out))
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.last(out)
        return out
