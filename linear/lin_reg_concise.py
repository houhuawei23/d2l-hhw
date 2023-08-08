import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

true_w = torch.tensor([2, -3.4])
true_b = 4.4

features, labels = d2l.synthetic_data(true_w, true_b, 1000)

def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

# Model
from torch import nn
net = nn.Sequential(nn.Linear(2, 1))

# Init params
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# Loss func
loss = nn.MSELoss()

# SGD
trainer = torch.optim.SGD(net.parameters(), lr = 0.03)

# Train
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

print(f'w的估计误差: {true_w - net[0].weight.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - net[0].bias}')
