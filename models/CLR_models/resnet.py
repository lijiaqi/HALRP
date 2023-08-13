import torch
import torch.nn as nn
import torch.nn.functional as F
from .clr import CLRBlock

__all__ = ["ResNet18_flat", "CLR_ResNet18", "batchNorm_CLR_ResNet18"]

class CLR_ResNet18(nn.Module):
    def __init__(self, currentModel, newClassifier):
        super(CLR_ResNet18, self).__init__()
        assert isinstance(
            currentModel, ResNet18_flat
        ), "The model class should be ResNet18_flat!"
        ### build self.features
        # conv_list = [0, 3, 6, 9, 12, 15, 18, 22, 25, 28, 31, 35, 38, 41, 44, 48, 51]
        self.clr_blocks_list = []
        new_layers = []
        for idx, layer in enumerate(currentModel.features):
            if isinstance(layer, nn.Conv2d) and layer.kernel_size != (1,1):
                clr_block = CLRBlock(layer, kernel_size=3)
                new_layers.append(clr_block)
                self.clr_blocks_list.append(clr_block)
            else:
                new_layers.append(layer)
        self.features = nn.Sequential(*new_layers)

        ### build blocks and shortcuts
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
        for i, j in enumerate(self.features):
            ### block0
            if i <= 2:
                self.block0.add_module(str(i), j)
            ### layer 1
            elif i >= 3 and i <= 7:
                self.block1.add_module(str(i), j)
            elif i == 8:
                self.shortcut1 = j
            elif i >= 9 and i <= 13:
                self.block2.add_module(str(i), j)
            elif i == 14:
                self.shortcut2 = j
            ### layer 2
            elif i >= 15 and i <= 19:
                self.block3.add_module(str(i), j)
            elif i >= 20 and i <= 21:
                self.shortcut3.add_module(str(i), j)
            elif i >= 22 and i <= 26:
                self.block4.add_module(str(i), j)
            elif i == 27:
                self.shortcut4 = j
            ### layer 3
            elif i >= 28 and i <= 32:
                self.block5.add_module(str(i), j)
            elif i >= 33 and i <= 34:
                self.shortcut5.add_module(str(i), j)
            elif i >= 35 and i <= 39:
                self.block6.add_module(str(i), j)
            elif i == 40:
                self.shortcut6 = j
            ### layer 4
            elif i >= 41 and i <= 45:
                self.block7.add_module(str(i), j)
            elif i >= 46 and i <= 47:
                self.shortcut7.add_module(str(i), j)
            elif i >= 48 and i <= 52:
                self.block8.add_module(str(i), j)
            elif i == 53:
                self.shortcut8 = j
        # head
        self.backbone_dim = currentModel.backbone_dim
        self.classifier = newClassifier

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
        logits = self.classifier(out)
        return logits

class batchNorm_CLR_ResNet18(nn.Module):
    def __init__(self, currentModel, BN, newClassifier):
        super(batchNorm_CLR_ResNet18, self).__init__()
        assert isinstance(
            currentModel, ResNet18_flat
        ), "The model class should be ResNet18_flat!"
        ### build self.features
        # conv_list = [0, 3, 6, 9, 12, 15, 18, 22, 25, 28, 31, 35, 38, 41, 44, 48, 51]
        # bn_list= [1, 4, 7, 10, 13, 16, 19, 21, 23, 26, 29, 32, 34, 36, 39, 42, 45, 47, 49, 52]
        self.clr_blocks_list = []
        new_layers = []
        for idx, layer in enumerate(currentModel.features):
            if isinstance(layer, nn.Conv2d) and layer.kernel_size != (1,1):
                clr_block = CLRBlock(layer, kernel_size=3)
                new_layers.append(clr_block)
                self.clr_blocks_list.append(clr_block)
            elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
                new_layers.append(BN[str(idx)])
            else:
                new_layers.append(layer)
        self.features = nn.Sequential(*new_layers)

        ### build blocks and shortcuts
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
        for i, j in enumerate(self.features):
            ### block0
            if i <= 2:
                self.block0.add_module(str(i), j)
            ### layer 1
            elif i >= 3 and i <= 7:
                self.block1.add_module(str(i), j)
            elif i == 8:
                self.shortcut1 = j
            elif i >= 9 and i <= 13:
                self.block2.add_module(str(i), j)
            elif i == 14:
                self.shortcut2 = j
            ### layer 2
            elif i >= 15 and i <= 19:
                self.block3.add_module(str(i), j)
            elif i >= 20 and i <= 21:
                self.shortcut3.add_module(str(i), j)
            elif i >= 22 and i <= 26:
                self.block4.add_module(str(i), j)
            elif i == 27:
                self.shortcut4 = j
            ### layer 3
            elif i >= 28 and i <= 32:
                self.block5.add_module(str(i), j)
            elif i >= 33 and i <= 34:
                self.shortcut5.add_module(str(i), j)
            elif i >= 35 and i <= 39:
                self.block6.add_module(str(i), j)
            elif i == 40:
                self.shortcut6 = j
            ### layer 4
            elif i >= 41 and i <= 45:
                self.block7.add_module(str(i), j)
            elif i >= 46 and i <= 47:
                self.shortcut7.add_module(str(i), j)
            elif i >= 48 and i <= 52:
                self.block8.add_module(str(i), j)
            elif i == 53:
                self.shortcut8 = j
        # head
        self.backbone_dim = currentModel.backbone_dim
        self.classifier = newClassifier

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
        logits = self.classifier(out)
        return logits

########################
# original Resnet18
########################
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
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(
        self,
        block,
        num_blocks,
        nf=32,
        num_cls=10,
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

        self.classifier = nn.Linear(
            nf * 8 * block.expansion * ss * ss, num_cls, bias=True
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
        out = F.relu(self.bn1(self.conv1(x.view(bsz, 3, self.img_sz, self.img_sz))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 2)

        out = out.view(out.size(0), -1)
        y = self.classifier(out)
        return y


def construct_flat_Res18(num_cls, nf=32, img_sz=32, track_bn_stats=False):
    def get_children(model: torch.nn.Module):
        # get children form model
        children = list(model.children())
        # print("children={}".format(children))
        flatt_children = []
        if children == []: # model=nn.Sequential() or model=Conv2D/Linear/...
            # if model has no children; model is last child! :O
            return [model]
        else:
            # look for children from children... to the last child!
            for child in children:
                # print("child={}".format(child))
                # try:
                flatt_children.extend(get_children(child))
                # except TypeError:
                #     flatt_children.append(get_children(child))
        return flatt_children
    temp_model = ResNet(
        block=BasicBlock,
        num_blocks=[2, 2, 2, 2],
        nf=nf,
        num_cls=num_cls,
        img_sz=img_sz,
        track_bn_stats=track_bn_stats,
    )
    # print("temp_model={}".format(temp_model))
    # print("list(temp_model.children())={}".format(list(temp_model.children())))
    new_model = nn.Sequential()
    for i, j in enumerate(get_children(temp_model)):
        # if i<= 40:
        new_model.add_module(str(i), j)
    # print("new_model={}".format(new_model))
    return new_model


# see strutures in "resnet18_flat.txt"
class ResNet18_flat(nn.Module):

    def __init__(self, num_cls, nf=32, img_sz=32, track_bn_stats=False):
        super(ResNet18_flat, self).__init__()
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
        self.classifier = nn.Sequential()

        resmodel = construct_flat_Res18(
            nf=nf,
            num_cls=num_cls,
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
            elif i >=18  and i <= 19:
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
            elif i >=29 and i <= 30:
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
                self.classifier = j
        self.backbone_dim = self.classifier.in_features

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
        out = self.classifier(out)
        return out
