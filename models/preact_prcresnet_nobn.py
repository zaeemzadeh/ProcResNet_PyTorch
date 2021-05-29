'''Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from utils import batch_norm
import numpy as np
# from .preact_resnet import PreActBottleneck


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
                # nn.BatchNorm2d(in_planes),
                nn.ReLU(),
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x
        out = self.conv1(F.relu(x))
        out = self.conv2(F.relu(out))
        out = self.conv3(F.relu(out))
        self.output = out + shortcut
        return self.output


class PreActProcResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, UoS=False):
        super(PreActProcResNet, self).__init__()
        self.in_planes = 64
        self.UoS = UoS

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)

        self.trans_conv1 = nn.Conv2d(64, 128, kernel_size=1, stride=2, padding=0, bias=False)
        self.in_planes *= 2
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=1)

        self.trans_conv2 = nn.Conv2d(128, 256, kernel_size=1, stride=2, padding=0, bias=False)
        self.in_planes *= 2
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=1)

        self.bn = nn.BatchNorm2d(64*block.expansion)
        if not self.UoS:
            self.linear = nn.Linear(64*block.expansion, num_classes)
        else:
            self.linear = nn.Linear(64*block.expansion, num_classes, bias=False)
            self.linear.weight.requires_grad = False
            nn.init.orthogonal_(self.linear.weight.data)
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
        with torch.no_grad():
            self.regularize_convs()
        self.input = x

        self.out_conv1 = self.conv1(self.input)
        out = self.layer1(self.out_conv1)

        self.out_trans1 = self.trans_conv1(out)
        out = self.layer2(self.out_trans1)

        self.out_trans2 = self.trans_conv2(out)
        # out = F.avg_pool2d(out, 2) * 4
        # self.out_trans2 = out.repeat(1,2,1,1)
        out = self.layer3(self.out_trans2)

        if self.UoS:
            # out = F.relu(self.bn(out))
            # out = self.bn(out)
            out = F.avg_pool2d(out, out.size(3))
            out_feat = out.view(out.size(0), -1)
            out = F.normalize(out_feat, dim=1, p=2)
            out = torch.abs(self.linear(out))
        else:
            # out = self.bn(out)
            out = F.relu(out)
            out = F.avg_pool2d(out, out.size(3))
            out_feat = out.view(out.size(0), -1)
            out = self.linear(out_feat)

        self.get_norms()
        return out, out_feat, self.norms

    def get_norms(self):
        with torch.no_grad():
            self.norms = []
            self.norms.append(batch_norm(self.input))
            self.norms.append(batch_norm(self.out_conv1))
            self.norms.extend([batch_norm(m.output) for m in self.layer1])
            self.norms.append(batch_norm(self.out_trans1))
            self.norms.extend([batch_norm(m.output) for m in self.layer2])
            self.norms.append(batch_norm(self.out_trans2))
            self.norms.extend([batch_norm(m.output) for m in self.layer3])
        return

    def regularize_convs(self, p=.5):
        if np.random.rand() > p:
            return

        # n_out, n_in, _, _ = self.conv1.weight.data.shape
        # self.conv1.weight.data = regularize_conv(self.conv1.weight.data, (32, 32), 3 * 1. * 10./9.)
        # self.conv1.weight.requires_grad = False

        n_out, n_in, _, _ = self.trans_conv1.weight.data.shape
        self.trans_conv1.weight.data = regularize_conv(self.trans_conv1.weight.data, (16, 16), 2. * 1.)
        # self.trans_conv1.weight.requires_grad = False

        n_out, n_in, _, _ = self.trans_conv2.weight.data.shape
        self.trans_conv2.weight.data = regularize_conv(self.trans_conv2.weight.data, (8, 8), 2. * 1.)
        # self.trans_conv2.weight.requires_grad = False

        return


def regularize_conv(kernel, input_shape, clip_to):

    kernel = kernel.permute((3, 2, 1, 0)) # tensor flow format

    kernel = torch_regularize_conv(kernel, input_shape, clip_to)

    kernel = kernel.permute((3, 2, 1, 0)) # back to PyTorch format
    return kernel

def torch_regularize_conv(kernel, input_shape, clip_to):
    A = torch.fft.fftn(kernel, s=input_shape, dim=(0, 1))
    # complex SVD using real formulation https://www.osti.gov/servlets/purl/756121
    A_resh = torch.reshape(A, (-1, A.shape[2], A.shape[3]))
    R_resh = A_resh.real
    I_resh = A_resh.imag

    upper_batch = torch.cat((R_resh, I_resh), dim=2)
    lower_batch = torch.cat((-I_resh, R_resh), dim=2)
    K_batch = torch.cat((upper_batch, lower_batch), dim=1)

    KTK = torch.matmul(K_batch.permute((0, 2, 1)), K_batch)

    _, sKTKinv = batch_sqrtm(KTK, numIters=20, reg=1.)


    K_tran_batch = torch.squeeze(torch.matmul(K_batch, sKTKinv)) * clip_to

    if len(K_tran_batch.shape) == 2:
        K_tran_batch = K_tran_batch[None, :, :]

    _, M, N = K_tran_batch.shape

    A_tran = K_tran_batch[:, 0:int(M / 2), 0:int(N / 2)] - 1j * K_tran_batch[:, 0:int(M / 2), -int(N / 2):]
    A_tran = torch.reshape(A_tran, A.shape).type(torch.complex64)

    clipped_kernel = torch.fft.ifftn(A_tran, dim=(0, 1)).real
    return clipped_kernel[np.ix_(*[range(d) for d in kernel.shape])]

def batch_sqrtm(A, numIters = 20, reg = 2.0):
    """
    Batch matrix root via Newton-Schulz iterations
    from: https://github.com/BorisMuzellec/EllipticalEmbeddings/blob/master/utils.py
    """
    batchSize = A.shape[0]
    dim = A.shape[1]
    #Renormalize so that the each matrix has a norm lesser than 1/reg, but only normalize when necessary
    normA = reg * torch.linalg.norm(A, axis=(1, 2))
    renorm_factor = torch.ones_like(normA)
    renorm_factor[torch.where(normA > 1.0)] = normA[torch.where(normA > 1.0)]
    renorm_factor = renorm_factor.reshape(batchSize, 1, 1)

    Y = torch.divide(A, renorm_factor)
    I = torch.eye(dim).to(Y.device).reshape(1, dim, dim).repeat(batchSize, 1, 1)
    Z = torch.eye(dim).to(Y.device).reshape(1, dim, dim).repeat(batchSize, 1, 1)
    for i in range(numIters):
        T = 0.5 * (3.0 * I - torch.matmul(Z, Y))
        Y = torch.matmul(Y, T)
        Z = torch.matmul(T, Z)
    sA = Y * torch.sqrt(renorm_factor)
    sAinv = Z / torch.sqrt(renorm_factor)
    return sA, sAinv



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

def PreActProcResNet164(UoS=False):
    return PreActProcResNet(PreActBottleneck, [18,18,18], UoS=UoS)

def PreActProcResNet11(UoS=False):
    return PreActProcResNet(PreActBottleneck, [1,1,1], UoS=UoS)


def test():
    net = PreActProcResNet164()
    y = net((torch.randn(1,3,32,32)))
    print(y.size())

# test()
