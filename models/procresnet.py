'''Pre-activation ProcResNet in PyTorch.

Reference:
Alireza Zaeemzadeh, Nazanin Rahnavard, and Mubarak Shah. 
"Norm-Preservation: Why Residual Networks Can Become Extremely Deep?." 
IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI) 2020
arXiv: 1805.07477
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import numpy as np
from .resnet import PreActBottleneck


def regularize_conv(kernel, input_shape, clip_to):
    kernel = kernel.permute((3, 2, 1, 0)) # tensor flow format
    A = torch.fft.fftn(kernel, s=input_shape, dim=(0, 1))
    # complex SVD using real formulation https://www.osti.gov/servlets/purl/756121
    A_resh = torch.reshape(A, (-1, A.shape[2], A.shape[3]))
    R_resh = A_resh.real
    I_resh = A_resh.imag

    upper_batch = torch.cat((R_resh, I_resh), dim=2)
    lower_batch = torch.cat((-I_resh, R_resh), dim=2)
    K_batch = torch.cat((upper_batch, lower_batch), dim=1)

    KTK = torch.matmul(K_batch.permute((0, 2, 1)), K_batch)

    _, sKTKinv = batch_sqrtm(KTK, numIters=10, reg=1.)


    K_tran_batch = torch.squeeze(torch.matmul(K_batch, sKTKinv)) * clip_to

    if len(K_tran_batch.shape) == 2:
        K_tran_batch = K_tran_batch[None, :, :]

    _, M, N = K_tran_batch.shape

    A_tran = K_tran_batch[:, 0:int(M / 2), 0:int(N / 2)] - 1j * K_tran_batch[:, 0:int(M / 2), -int(N / 2):]
    A_tran = torch.reshape(A_tran, A.shape).type(torch.complex64)

    clipped_kernel = torch.fft.ifftn(A_tran, dim=(0, 1)).real
    clipped_kernel = clipped_kernel[np.ix_(*[range(d) for d in kernel.shape])]
    clipped_kernel = clipped_kernel.permute((3, 2, 1, 0)) # back to PyTorch format
    return clipped_kernel

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
    
class ProcResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ProcResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, num_blocks, stride=1)

        self.trans_conv1 = nn.Conv2d(64, 128, kernel_size=1, stride=2, padding=0, bias=False)
        self.in_planes *= 2
        self.layer2 = self._make_layer(block, 32, num_blocks, stride=1)

        self.trans_conv2 = nn.Conv2d(128, 256, kernel_size=1, stride=2, padding=0, bias=False)
        self.in_planes *= 2
        self.layer3 = self._make_layer(block, 64, num_blocks, stride=1)

        self.bn = nn.BatchNorm2d(64*block.expansion)
        self.linear = nn.Linear(64*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # if self.training:
        #     self.regularize_convs(p=self.regul_freq)
        self.input = x

        self.out_conv1 = self.conv1(self.input)
        out = self.layer1(self.out_conv1)

        self.out_trans1 = self.trans_conv1(out)
        out = self.layer2(self.out_trans1)

        self.out_trans2 = self.trans_conv2(out)
        out = self.layer3(self.out_trans2)

        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, out.size(3))
        out_feat = out.view(out.size(0), -1)
        out = self.linear(out_feat)

        return out

    def regularize_convs(self, p=0.5):
        if np.random.rand() > p:
            return
    
        with torch.no_grad():
            n_out, n_in, _, _ = self.conv1.weight.data.shape
            self.conv1.weight.data = regularize_conv(self.conv1.weight.data, (32, 32), np.sqrt(float(n_out)/n_in))

            n_out, n_in, _, _ = self.trans_conv1.weight.data.shape
            self.trans_conv1.weight.data = regularize_conv(self.trans_conv1.weight.data, (16, 16), np.sqrt(float(n_out)/n_in))

            n_out, n_in, _, _ = self.trans_conv2.weight.data.shape
            self.trans_conv2.weight.data = regularize_conv(self.trans_conv2.weight.data, (8, 8), np.sqrt(float(n_out)/n_in))

        return

class ProcResNet166(ProcResNet):
    def __init__(self, n_class=10):
        super(ProcResNet166, self).__init__(PreActBottleneck, 18, n_class)

class ProcResNet274(ProcResNet):
    def __init__(self, n_class=10):
        super(ProcResNet274, self).__init__(PreActBottleneck, 30, n_class)

class ProcResNet634(ProcResNet):
    def __init__(self, n_class=10):
        super(ProcResNet634, self).__init__(PreActBottleneck, 70, n_class)

class ProcResNet1003(ProcResNet):
    def __init__(self, n_class=10):
        super(ProcResNet1003, self).__init__(PreActBottleneck, 111, n_class)
