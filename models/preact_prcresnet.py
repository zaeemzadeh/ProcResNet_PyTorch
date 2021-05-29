'''Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import batch_norm


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


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
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        self.output = out + shortcut
        return self.output


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv1.register_forward_hook(hook_fn)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64*block.expansion, num_classes)
        self.norms = []

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            # layers[-1].register_forward_hook(hook_fn)
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        self.input = x
        self.out_conv1 = self.conv1(self.input)
        out = self.layer1(self.out_conv1)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size(3))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        self.get_norms()
        return out, self.norms

    def get_norms(self):
        with torch.no_grad():
            self.norms = []
            self.norms.append(batch_norm(self.input))
            self.norms.append(batch_norm(self.out_conv1))
            self.norms.extend([batch_norm(m.output) for m in self.layer1])
            self.norms.extend([batch_norm(m.output) for m in self.layer2])
            self.norms.extend([batch_norm(m.output) for m in self.layer3])
        return


def hook_fn(m, i, o):
  # print(m)
  # print("------------Input------------")
  # print(i[0].shape)
  # print("------------Output ------------")
  print(o.shape, o[0,0,0,0])


# def PreActResNet18():
#     return PreActResNet(PreActBlock, [2,2,2,2])
#
# def PreActResNet34():
#     return PreActResNet(PreActBlock, [3,4,6,3])
#
# def PreActResNet50():
#     return PreActResNet(PreActBottleneck, [3,4,6,3])
#
# def PreActResNet101():
#     return PreActResNet(PreActBottleneck, [3,4,23,3])
#
# def PreActResNet152():
#     return PreActResNet(PreActBottleneck, [3,8,36,3])

def PreActResNet164():
    return PreActResNet(PreActBottleneck, [18,18,18])

def PreActResNet11():
    return PreActResNet(PreActBottleneck, [1,1,1])


def test():
    net = PreActResNet164()
    y = net((torch.randn(1,3,32,32)))
    print(y.size())

# test()
