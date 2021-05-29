'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from importlib import import_module

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import inspect

import os
import sys
import argparse
import shutil

from models import *
from utils import progress_bar, count_parameters, count_convs, create_result_dir, Logger

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if callable(getattr(net.module, "regularize_convs", None)):
            net.module.regularize_convs(p=args.regul_freq)

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | LR: %.3f'
                    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, optimizer.param_groups[0]['lr']))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving.. best accuracy: ', acc)
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, os.path.join( result_dir, 'ckpt.pth' ))
        best_acc = acc


parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--model_file', type=str, default='models/procresnet.py')
parser.add_argument('--model_name', type=str, default='ProcResNet166')
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--regul_freq', type=float, default=0.5)
# Train settings
parser.add_argument('--batchsize', type=int, default=128)  
parser.add_argument('--training_epoch', type=int, default=300) 
parser.add_argument('--initial_lr', type=float, default=0.1)  
parser.add_argument('--lr_decay_epoch', type=float, nargs='*', default=[150, 225])  
parser.add_argument('--lr_decay_rate', type=float, default=0.1)
parser.add_argument('--weight_decay', type=float, default=0.0001)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
# Data loading code
normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                 std=[x/255.0 for x in [63.0, 62.1, 66.7]])


transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: F.pad(
                        Variable(x.unsqueeze(0), requires_grad=False, volatile=True),
                        (4,4,4,4),mode='reflect').data.squeeze()),
    transforms.ToPILImage(),
    transforms.RandomCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
    ])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize
    ])


kwargs = {'num_workers': 1, 'pin_memory': True}
assert(args.dataset == 'cifar10' or args.dataset == 'cifar100')
trainloader = torch.utils.data.DataLoader(
    datasets.__dict__[args.dataset.upper()]('./data', train=True, download=True,
                        transform=transform_train),
    batch_size=args.batchsize, shuffle=True, **kwargs)
testloader = torch.utils.data.DataLoader(
    datasets.__dict__[args.dataset.upper()]('./data', train=False, transform=transform_test),
    batch_size=args.batchsize, shuffle=True, **kwargs)

# Model
print('==> Building model..')
ext = os.path.splitext(args.model_file)[1]
mod_path = '.'.join(os.path.split(args.model_file.replace(ext, '')))
mod = import_module(mod_path)
net = getattr(mod, args.model_name)()
net = net.to(device)

print('Number of parameters: ', count_parameters(net)/1e6, 'M')
print('Number of convs in the model: ', count_convs(net))

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.initial_lr,
                      momentum=0.9, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay_epoch, gamma=args.lr_decay_rate)
# create result dir
result_dir = create_result_dir(args.model_name)
shutil.copy(args.model_file, os.path.join(result_dir, os.path.basename(args.model_file)))

sys.stdout = Logger(os.path.join(result_dir, os.path.basename('log')))
print(result_dir)
print(args)

for epoch in range(start_epoch, start_epoch+args.training_epoch):
    train(epoch)
    test(epoch)
    scheduler.step()
