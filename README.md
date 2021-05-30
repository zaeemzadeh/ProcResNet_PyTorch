# Procrustes ResNet: ResNet with Norm-Preserving Transition Blocks
Pytorch implementation of Procrustes ResNet (ProcResNet) proposed in: 

Zaeemzadeh, Alireza, Nazanin Rahnavard, and Mubarak Shah. 
"Norm-Preservation: Why Residual Networks Can Become Extremely Deep?." 
IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI) 2020 
[link](https://arxiv.org/pdf/1805.07477.pdf)

- **Note:** For the original impementation using [Chainer](https://chainer.org/), see [here](https://github.com/zaeemzadeh/ProcResNet).


## Requirements

Tested on:
- Python 3.9.2
- cuda 11.2
- torch 1.8.1
- torchvision 0.9.1
- numpy 1.20.1

## Quick Start

Sample commands:

```bash
python main.py --model_file 'models/procresnet.py' --model_name 'ProcResNet166' --regul_freq 0.5 --batchsize 128 --training_epoch 300 --lr_decay_epoch 150 225 --initial_lr 0.1 --dataset 'cifar10'

python main.py --model_file 'models/procresnet.py' --model_name 'ProcResNet274' --dataset 'cifar10'

python main.py --model_file 'models/resnet.py'     --model_name 'ResNet272'     --dataset 'cifar10'

python main.py --model_file 'models/procresnet.py' --model_name 'ProcResNet274' --dataset 'cifar100'

python main.py --model_file 'models/resnet.py'     --model_name 'ResNet272'     --dataset 'cifar100'
```

'regul_freq' is a number in range [0, 1] and determines how often the regularization is performed.

## About Regularization of the Conv Layers
The ProcResNet class has a method called 'regularize_convs', which is called after gradient descent update to enforce norm-preservation on the transition blocks.

See the details at `regularize_convs` function in `models/procresnet.py`.

Gradient norm ratio for ResNet (top) and ProcResNet (bottom):

<img src="https://github.com/zaeemzadeh/ProcResNet/blob/master/imgs/animated.gif" width="480">




## Citing This Work
If you find this work useful, please use the following BibTeX entry.
```
@article{zaeemzadeh2018norm,
  title={Norm-Preservation: Why Residual Networks Can Become Extremely Deep?},
  author={Zaeemzadeh, Alireza and Rahnavard, Nazanin and Shah, Mubarak},
  journal = {Pattern Analysis and Machine Intelligence, IEEE Transactions on},
  year = {2020}
}

```



