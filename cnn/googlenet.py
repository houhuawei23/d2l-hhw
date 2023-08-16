import os

import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# import sys
# sys.path.append("..")
# from tensorkit import network
# from ..tensorkit import network
from tensorkit import network
from tensorkit.datasets.load import load_fashion_mnist
BASE_DIR = os.environ.get('BASE_DIR')

class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        
        self.p1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)
        
    def forward(self, x):
        p1 = F.relu(self.p1(x))
        # print(f"p1 out:\t{p1.shape}")
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        # print(f"p2 out:\t{p2.shape}") 
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        # print(f"p3 out:\t{p3.shape}")
        p4 = F.relu(self.p4_2(F.relu(self.p4_1(x))))
        # print(f"p4 out:\t{p4.shape}")
        return torch.cat((p1, p2, p3, p4), dim=1)

# GoogLeNet
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

b2 = nn.Sequential(
    nn.Conv2d(64, 64, kernel_size=1),
    nn.ReLU(),
    nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

b3 = nn.Sequential(
    Inception(192, 64, (96, 128), (16, 32), 32),
    Inception(256, 128, (128, 192), (32, 96), 64),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

b4 = nn.Sequential(
    Inception(480, 192, (96, 208), (16, 48), 64),
    Inception(512, 160, (112, 224), (24, 64), 64),
    Inception(512, 128, (128, 256), (24, 64), 64),
    Inception(512, 112, (144, 288), (32, 64), 64),
    Inception(528, 256, (160, 320), (32, 128), 128),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

b5 = nn.Sequential(
    Inception(832, 256, (160, 320), (32, 128), 128),
    Inception(832, 384, (192, 384), (48, 128), 128),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten()
)

googlenet = nn.Sequential(
    b1, b2, b3, b4, b5,
    nn.Linear(1024, 10)
)

if __name__ == '__main__':
    # X = torch.rand(size=(5, 1, 96, 96))
    # for layer in net:
    #     X = layer(X)
    #     print(layer.__class__.__name__, 'oupt shape:\t', X.shape)
    """
    Sequential oupt shape:   torch.Size([5, 64, 24, 24])
    Sequential oupt shape:   torch.Size([5, 192, 12, 12])
    Sequential oupt shape:   torch.Size([5, 480, 6, 6])
    Sequential oupt shape:   torch.Size([5, 832, 3, 3])
    Sequential oupt shape:   torch.Size([5, 1024])
    Linear oupt shape:       torch.Size([5, 10])    
    """

    # inception = Inception(3, 64, (96, 128), (16, 32), 32)

    # X = torch.rand(size=(5, 3, 224, 224))
    # Y = inception(X)
    # print(f"out:\t{Y.shape}")
    """
    p1 out: torch.Size([5, 64, 224, 224])
    p2 out: torch.Size([5, 128, 224, 224])
    p3 out: torch.Size([5, 32, 224, 224])
    p4 out: torch.Size([5, 32, 224, 224])
    out:    torch.Size([5, 256, 224, 224])
    """
    
    save_dir = os.path.join(BASE_DIR, 'd2l-hhw', 'cnn', 'results')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    lr, num_epochs, batch_size = 0.1, 10, 128
    train_iter, test_iter = load_fashion_mnist(batch_size, resize=96)
    from tensorkit.network import train
    from tensorkit.utils import try_gpu
    train(googlenet, train_iter, test_iter, 
          num_epochs, lr, device=try_gpu(0), 
          save_dir=save_dir, fname="googlenet.png")
    