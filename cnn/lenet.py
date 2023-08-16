import os

from tensorkit.network import LeNet
from tensorkit.train import train
from tensorkit.datasets.load import load_mnist
from tensorkit.utils import try_gpu

BASE_DIR = os.environ.get('BASE_DIR')


save_dir = os.path.join(BASE_DIR, 'd2l-hhw', 'cnn', 'results')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
lr, num_epochs, batch_size = 0.1, 10, 64
train_iter, test_iter = load_mnist(batch_size)

train(LeNet, train_iter, test_iter, 
      num_epochs, lr, device=try_gpu(0), 
      save_dir=save_dir, fname="lenet.png")



