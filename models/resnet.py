'''Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.ReLU(),
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        self.output = out + shortcut
        return self.output


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        # self.conv1.register_forward_hook(hook_fn)
        self.layer1 = self._make_layer(block, 16, num_blocks, stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks, stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks, stride=2)
        self.bn2 = nn.BatchNorm2d(64*block.expansion)
        self.linear = nn.Linear(64*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        self.input = x
        self.out_conv1 = F.relu(self.bn1(self.conv1(self.input)))
        out = self.layer1(self.out_conv1)
        out = self.layer2(out)
        out = self.layer3(out)
        
        out = F.relu(self.bn2(out))
        out = F.avg_pool2d(out, out.size(3))
        out_feat = out.view(out.size(0), -1)
        out = self.linear(out_feat)
        return out


class ResNet164(ResNet):
    def __init__(self, n_class=10):
        super(ResNet164, self).__init__(PreActBottleneck, 18, n_class)

class ResNet272(ResNet):
    def __init__(self, n_class=10):
        super(ResNet272, self).__init__(PreActBottleneck, 30, n_class)

class ResNet632(ResNet):
    def __init__(self, n_class=10):
        super(ResNet632, self).__init__(PreActBottleneck, 70, n_class)

class ResNet1001(ResNet):
    def __init__(self, n_class=10):
        super(ResNet1001, self).__init__(PreActBottleneck, 111, n_class)
