import torch
from d2l import torch as d2l

"""
softmax Model
y = softmax(Wx + b)
input: tensor(28*28)
output: 10
"""

# 1 Load Data
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 2 Init Params
num_inputs = 28 * 28
num_outputs = 10

W = torch.normal(0, 0.1, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_inputs, requires_grad=True)

# softmax func


def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X / partition

# 3 Define Net


def net(X):
    # 为啥这里是XW，我记得数学上是WX？？
    return softmax(torch.matmul(X, W) + b)

# 4 Cross Entropy


def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])


def accuracy(y_hat, y):
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum)


y = torch.tensor([0, 2])
y_hat = torch.tensor(
    [[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]]
)

print(cross_entropy(y_hat, y))


