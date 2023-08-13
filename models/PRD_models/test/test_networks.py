import torch
import torch.nn as nn

def test_SupConLeNet():
    from PRD_models import SupConLeNet
    model = SupConLeNet(input_size=32, in_channel=3, num_layers=3)
    print("SupConLeNet:", model)
    dummy = torch.rand(size=(10,3,32,32))
    print("output shape=", model.forward(dummy).shape)


def test_SupConLeNet_bnFalse():
    from PRD_models import SupConLeNet

    model = SupConLeNet(input_size=32, in_channel=3, num_layers=3, track_bn_stats=False)
    print("SupConLeNet (bnFalse):", model)
    dummy = torch.rand(size=(10, 3, 32, 32))
    print("output shape=", model.forward(dummy).shape)


def test_SupConWideLeNet():
    from PRD_models import SupConWideLeNet
    model = SupConWideLeNet(input_size=32, in_channel=1, num_layers=3)
    print("SupConWideLeNet:", model)
    dummy = torch.rand(size=(10,1,32,32))
    print("output shape=", model.forward(dummy).shape)


def test_SupConWideLeNet_bnFalse():
    from PRD_models import SupConWideLeNet

    model = SupConWideLeNet(
        input_size=32, in_channel=1, num_layers=3, track_bn_stats=False
    )
    print("SupConWideLeNet (bnFalse):", model)
    dummy = torch.rand(size=(10, 1, 32, 32))
    print("output shape=", model.forward(dummy).shape)


def test_SupConAlexNet():
    from PRD_models import SupConAlexNet
    model = SupConAlexNet(input_size=32, num_layers=3)
    print("SupConAlexNet:", model)
    dummy = torch.rand(size=(10,3,32,32))
    print("output shape=", model.forward(dummy).shape)


def test_SupConAlexNet_bnFalse():
    from PRD_models import SupConAlexNet

    model = SupConAlexNet(input_size=32, num_layers=3, track_bn_stats=False)
    print("SupConAlexNet (bnFalse):", model)
    dummy = torch.rand(size=(10, 3, 32, 32))
    print("output shape=", model.forward(dummy).shape)


def test_SupConResNet18():
    from PRD_models import SupConResNet18

    model = SupConResNet18(input_size=32, num_layers=3)
    print("SupConResNet18:", model)
    dummy = torch.rand(size=(10, 3, 32, 32))
    print("output shape=", model.forward(dummy).shape)


def test_SupConResNet18_bnFalse():
    from PRD_models import SupConResNet18

    model = SupConResNet18(input_size=32, num_layers=3, track_bn_stats=False)
    print("SupConResNet18 (bnFalse):", model)
    dummy = torch.rand(size=(10, 3, 32, 32))
    print("output shape=", model.forward(dummy).shape)
