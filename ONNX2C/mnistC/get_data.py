import torch, torchvision
from torch import nn,optim
from torch.autograd import Variable as var

#train_data = torchvision.datasets.MNIST('data',train=True,download=True,transform=T)
val_data = torchvision.datasets.MNIST("", download=True)
