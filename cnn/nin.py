import torch
from torch import nn


def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())
    
nin_net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(3, stride=2),
    nn.Dropout(0.5),
    # 标签类别数是10
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    nn.AdaptiveAvgPool2d((1, 1)),
    # 将四维的输出转成二维的输出，其形状为(批量大小,10)
    nn.Flatten())

from tensorkit.train import train
from tensorkit.datasets.load import load_fashion_mnist
from tensorkit.utils import try_gpu

X = torch.rand(size=(1, 1, 224, 224))
for layer in nin_net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
    
lr, num_epochs, batch_size = 0.1, 10, 64
train_iter, test_iter = load_fashion_mnist(batch_size, resize=224)

train(nin_net, train_iter, test_iter, num_epochs, lr, device=try_gpu(0))