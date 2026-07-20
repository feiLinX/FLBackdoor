import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ResNet18(nn.Module):
    """For 224x224 inputs — standard ImageNet-style stem (7x7 conv stride 2 + maxpool)."""
    def __init__(self):
        super(ResNet18, self).__init__()
        base = torchvision.models.resnet18()  # fresh instance, independent weights

        self.net = nn.Sequential(*list(base.children())[:-1], # [b, 512, 1, 1]
                                nn.Flatten(), # [b, 512, 1, 1] => [b, 512]
                                nn.Linear(512, 10)
                                )

    def forward(self, x, return_features=False):
        x = self.net(x)
        features = x.view(x.size(0), -1)
        if return_features:
            return x, features
        else:
            return x
        

class ResNet18Small(nn.Module):
    """For 32x32 inputs — CIFAR-style stem (3x3 conv stride 1, no maxpool).

    The ImageNet stem downsamples a 32x32 image to 8x8 before layer1 even
    runs, and all the way to 1x1 by layer4, wasting most of layer3/layer4's
    capacity on a 1-2 pixel feature map. Replacing conv1/maxpool keeps the
    spatial resolution reasonable through all four residual stages.
    """
    def __init__(self):
        super(ResNet18Small, self).__init__()
        base = torchvision.models.resnet18()  # fresh instance, independent weights

        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(base.conv1.weight, mode='fan_out', nonlinearity='relu')  # match torchvision's own init scheme
        base.maxpool = nn.Identity()

        self.net = nn.Sequential(*list(base.children())[:-1], # [b, 512, 1, 1]
                                nn.Flatten(), # [b, 512, 1, 1] => [b, 512]
                                nn.Linear(512, 10)
                                )

    def forward(self, x, return_features=False):
        x = self.net(x)
        features = x.view(x.size(0), -1)
        if return_features:
            return x, features
        else:
            return x
        

class ResNet34(nn.Module):
    """For 224x224 inputs — standard ImageNet-style stem (7x7 conv stride 2 + maxpool)."""
    def __init__(self):
        super(ResNet34, self).__init__()
        base = torchvision.models.resnet34()  # fresh instance, independent weights

        self.net = nn.Sequential(*list(base.children())[:-1], # [b, 512, 1, 1]
                                nn.Flatten(), # [b, 512, 1, 1] => [b, 512]
                                nn.Linear(512, 10)
                                )

    def forward(self, x, return_features=False):
        x = self.net(x)
        features = x.view(x.size(0), -1)
        if return_features:
            return x, features
        else:
            return x


class ResNet34Small(nn.Module):

    def __init__(self):
        super(ResNet34Small, self).__init__()
        base = torchvision.models.resnet34()  # fresh instance, independent weights

        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(base.conv1.weight, mode='fan_out', nonlinearity='relu')  # match torchvision's own init scheme
        base.maxpool = nn.Identity()

        self.net = nn.Sequential(*list(base.children())[:-1], # [b, 512, 1, 1]
                                nn.Flatten(), # [b, 512, 1, 1] => [b, 512]
                                nn.Linear(512, 10)
                                )

    def forward(self, x, return_features=False):
        x = self.net(x)
        features = x.view(x.size(0), -1)
        if return_features:
            return x, features
        else:
            return x
        

class ResNet50(nn.Module):
    """For 224x224 inputs — standard ImageNet-style stem (7x7 conv stride 2 + maxpool)."""
    def __init__(self, pretrained=False):
        super(ResNet50, self).__init__()
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        base = torchvision.models.resnet50(weights=weights)  # from-scratch unless pretrained=True

        self.net = nn.Sequential(*list(base.children())[:-1], # [b, 2048, 1, 1]
                                nn.Flatten(), # [b, 2048, 1, 1] => [b, 2048]
                                nn.Linear(2048, 10)
                                )

    def forward(self, x, return_features=False):
        x = self.net(x)
        features = x.view(x.size(0), -1)
        if return_features:
            return x, features
        else:
            return x
        

class ResNet50Small(nn.Module):

    def __init__(self, pretrained=False):
        super(ResNet50Small, self).__init__()
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        base = torchvision.models.resnet50(weights=weights)  # layer1-4 pretrained if requested; conv1 is replaced below regardless

        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(base.conv1.weight, mode='fan_out', nonlinearity='relu')  # match torchvision's own init scheme
        base.maxpool = nn.Identity()

        self.net = nn.Sequential(*list(base.children())[:-1], # [b, 2048, 1, 1]
                                nn.Flatten(), # [b, 2048, 1, 1] => [b, 2048]
                                nn.Linear(2048, 10)
                                )

    def forward(self, x, return_features=False):
        x = self.net(x)
        features = x.view(x.size(0), -1)
        if return_features:
            return x, features
        else:
            return x
