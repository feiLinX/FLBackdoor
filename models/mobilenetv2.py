import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


__all__ = ['MobileNetV2']

class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out
    

class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.classifier = nn.Linear(1280, num_classes)
        self.avg_pool2d = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False, return_logits=True):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        # out = F.avg_pool2d(out, 4)
        out = self.avg_pool2d(out)

        features = out.view(out.size(0), -1)

        if not return_logits:
            return features

        out = self.classifier(features)
        if return_features:
            return out, features
        else:
            return out


class MobileNetV2Large(nn.Module):
    """For 224x224 inputs — standard ImageNet MobileNetV2 (stride-2 stem + stride-2 stages).

    Unlike MobileNetV2 above (whose stem/stage-2 strides were relaxed to 1
    for 32x32 CIFAR-style inputs), this uses torchvision's unmodified
    ImageNet MobileNetV2, matching the ResNet18/ResNet34/ResNet50 Large
    variants elsewhere in this notebook.
    """
    def __init__(self, pretrained=False):
        super(MobileNetV2Large, self).__init__()
        weights = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
        base = torchvision.models.mobilenet_v2(weights=weights)  # from-scratch unless pretrained=True

        self.features = base.features  # [b, 1280, 7, 7] for 224x224 input
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(1280, 10)

    def forward(self, x, return_features=False):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        f = x.view(x.size(0), -1)
        if return_features:
            return x, f
        else:
            return x
