import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


def test_CLRBlock():
    from CLR_models.clr import CLRBlock

    conv = nn.Conv2d(32, 64, kernel_size=3)
    print("naive conv=", conv)
    clrblock = CLRBlock(conv)
    print("clrblock =", clrblock)
    assert conv is clrblock.original_conv


def test_CLR_AlexNet():
    from CLR_models.alexnet import CLR_AlexNet, AlexNet

    alex = AlexNet(num_cls=10, img_sz=32, track_bn_stats=True)

    ### create first CLR model
    newHead = nn.Linear(alex.backbone_dim, 10)
    clr_alex = CLR_AlexNet(alex, newHead)
    print(clr_alex)
    dummy = torch.rand(size=(12, 3, 32, 32))
    out = clr_alex(dummy)
    print(out.shape)

    ### create second CLR model
    newHead = nn.Linear(alex.backbone_dim, 10)
    clr_alex_2 = CLR_AlexNet(alex, newHead)
    # [0,5,10] is Conv layers in AlexNet.features
    conv_list = [0, 5, 10]
    for idx in range(len(clr_alex.features)):
        if idx in conv_list:
            assert (
                clr_alex.features[idx].original_conv
                is clr_alex_2.features[idx].original_conv
            )
            assert (
                clr_alex.features[idx].clr_conv is not clr_alex_2.features[idx].clr_conv
            )
        else:
            assert clr_alex.features[idx] is clr_alex_2.features[idx]


def test_ResNet18_flat():
    from CLR_models.resnet import ResNet18_flat

    flat_res18 = ResNet18_flat(nf=32, num_cls=10, img_sz=32, track_bn_stats=False)
    print(flat_res18)
    dummy = torch.rand(size=(10, 3, 32, 32))
    out = flat_res18(dummy)
    assert out.shape == (10, 10)

    # conv_list = [0, 3, 6, 9, 12, 15, 18, 22, 25, 28, 31, 35, 38, 41, 44, 48, 51]
    conv_list = []
    for idx, layer in enumerate(flat_res18.features):
        if isinstance(layer, nn.Conv2d) and layer.kernel_size != (1, 1):
            conv_list.append(idx)
    print("conv_list=", conv_list)

    # bn_list= [1, 4, 7, 10, 13, 16, 19, 21, 23, 26, 29, 32, 34, 36, 39, 42, 45, 47, 49, 52]
    bn_list = []
    for idx, layer in enumerate(flat_res18.features):
        if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
            bn_list.append(idx)
    print("bn_list=", bn_list)


def test_CLR_ResNet18():
    from CLR_models.resnet import ResNet18_flat, CLR_ResNet18

    res18 = ResNet18_flat(nf=32, num_cls=10, img_sz=32, track_bn_stats=False)

    ### create first CLR model
    newHead = nn.Linear(res18.backbone_dim, 10)
    clr_res18 = CLR_ResNet18(res18, newHead)
    print(clr_res18)

    dummy = torch.rand(size=(10, 3, 32, 32))
    out = clr_res18(dummy)
    assert out.shape == (10, 10)

    ### create second CLR model
    newHead = nn.Linear(res18.backbone_dim, 10)
    clr_res18_2 = CLR_ResNet18(res18, newHead)
    conv_list = [0, 3, 6, 9, 12, 15, 18, 22, 25, 28, 31, 35, 38, 41, 44, 48, 51]
    for idx in range(len(clr_res18_2.features)):
        if idx in conv_list:
            assert (
                clr_res18.features[idx].original_conv
                is clr_res18_2.features[idx].original_conv
            )
            assert (
                clr_res18.features[idx].clr_conv is not clr_res18_2.features[idx].clr_conv
            )
        else:
            assert clr_res18.features[idx] is clr_res18_2.features[idx]