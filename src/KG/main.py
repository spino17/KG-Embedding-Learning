from KG.architechures import Compositional, HOLE, ComplEx
from KG.models import Network
import torch
from torch import nn
import numpy as np

# declaring hyperparameters
num_dim = 5  # dimension of embedding vector
num_entities = 20  # number of words in vocabulary
num_relations = 10  # number of relations
alpha = 1  # coefficient of regularization term
batch_size = 2  # backprop for this many combined datapoints
num_epochs = 3  # number of loops over training dataset

# data preprocessing
# TODO

# defining the model
model = Network(HOLE(num_dim, num_entities, num_relations))
model_ = Network(ComplEx(num_dim, num_entities, num_relations))
print(model.model)
g = torch.randn(2, 20)
h = torch.randn(2, 20)
j = torch.randn(2, 10)
print(type(g), type(h), type(j))
output = model.model(g, h, j)
print(output)
output_ = model_.model(g, h, j)
print(output_)
# compiling the model
model.compile(optimizer_name="adam", loss="LogisticLoss", regularizer="L2")

x_train = np.array([[1, 2, 3], [3, 4, 5], [5, 6, 6], [1, 2, 1]])
print(x_train.shape)
y_train = np.array([[1], [0], [1], [1]])
print(y_train.shape)
z = torch.from_numpy(y_train)
b = torch.from_numpy(x_train)
print(b.type())
# training call
print(model.predict(x_train))
model.fit(x_train, y_train, batch_size, num_epochs, validation_split=0)
print(model.model)
print(model.optimizer)

# performance evaluation and predictions on testing dataset


from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

x = torch.randn(10, 4)
y = torch.randn(10, 1)
dataset = TensorDataset(x, y)
lengths = lengths = [int(len(dataset) * 0.8), int(len(dataset) * 0.2)]
subset1, subset2 = random_split(dataset, lengths)
print(subset1, subset2)
loader = DataLoader(subset1, batch_size=2)
for batch_ndx, sample in enumerate(loader):
    print(sample[0].size())
    # print(sample[1])

import torch

x = torch.Tensor([[1, 1, 1], [2, 2, 2]])
# y = nn.Tensor([[5, 5, 5], [6, 6, 6]])
print(torch.matmul(torch.transpose(x, 0, 1), y))
print(x.size())
y = nn.Linear(3, 3, 2)
print(y)
print(y[1])
z = torch.transpose(y, 0, 1)
print(z.size())
print(x.size())
p = torch.matmul(x, y)
print(p)
z = y(x)
print(z.size())


y = nn.Linear(num_entity, num_dim, bias=Falses)

x = torch.randn(2, 5)
y = nn.Linear(5, 3, bias=False)
w = list(y.parameters())
print(w)
t = torch.eye(2, 5)
s = y(t)
print(s.size())
print(torch.transpose(s, 0, 1).size())
y_ = nn.Linear(5, 3, bias=False)
w_ = list(y_.parameters())
print(w_)
t_ = torch.eye(2, 5)
s_ = y_(t_)
print(s_)

t = torch.transpose(s, 0, 1)
print(t)
tp = diag(tp).view(-1, num_dim, num_dim)

b = torch.mm(s, torch.transpose(s_, 0, 1))
print(b)
c = torch.diag(b).view(-1, 1)
print(c.size())

# s, s_ are embeddings
e = s * s_
print(e)
print(s)
import numpy as np

s = s.detach().numpy()
s = np.roll(s, 2, axis=1)
s = torch.from_numpy(s)
s = s.clone().detach().requires_grad_(True)
print(s)
n = torch.mean(s, dim=1)
print(n)

q = torch.empty(2, 1)
print(q)
m = torch.cat([q, s_], dim=1)
print(m)
print(s_)

print(torch.cuda.is_available())

from torch.utils.data import DataLoader, Dataset, TensorDataset

x = torch.randn(10, 20)
y = torch.randn(10, 20)
z = torch.randn(10, 5)
w = torch.randn(10, 1)
dataset = TensorDataset(x, y, z, w)
print(type(dataset))
loader = DataLoader(dataset, -1)
for batch_ndx, sample in enumerate(loader):
    print(batch_ndx)
    print(sample[2].shape)

x = [[1, 2], [3, 4], [6, 7]]
y = [[3, 5], [1, 2], [8, 9]]
x = np.array(x)
print(x[:, -1])
dataset = TensorDataset(x, y)

from torch.nn.functional import one_hot

indices = torch.randint(0, 5, size=(4, 1))
print(indices)
o = one_hot(indices, 5).view(-1, 5)
print(o.size())
print(o)


l1 = nn.Linear(3, 3)
l1.weight.data.fill_(0)
l1.bias.data.fill_(0)
x = torch.ones(2, 3)
y = l1(x)

# backward one loss only
loss1 = (y - 1).abs().sum()
loss1.backward(retain_graph=True)

# l1.weight.grad = None
loss2 = (y + 1).abs().sum()
loss2.backward()
print(l1.weight.grad)

# backward both losses together
l1.weight.grad = None
loss1 = (y - 1).abs().sum()
loss2 = (y + 1).abs().sum()
(loss1 + loss2).backward()
print(l1.weight.grad)

loss = (y - 1).abs().sum() + (y + 1).abs().sum()
loss.backward(retain_graph=True)
print(l1.weight.grad)

import numpy as np
np.kron(np.eye(3), np.ones((2, 2)))

import torch
def kron(a, b):
    """
    Kronecker product of matrices a and b with leading batch dimensions.
    Batch dimensions are broadcast. The number of them mush
    :type a: torch.Tensor
    :type b: torch.Tensor
    :rtype: torch.Tensor
    """
    siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
    res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
    siz0 = res.shape[:-4]
    return res.reshape(siz0 + siz1)

print(kron(torch.eye(2), torch.ones(3, 3)))

x = torch.randn(5, 3)
print(x)
print(x.unsqueeze(-1).size())

def single_tensor_product(a, b, dim=1):
    # TODO
q = np.array([[1, 2], [2, 4]])
x = q[
print(q.size)

x = torch.randn(2, 10)
y = x.long()
print(x, y)
print(np.type(y))
y = x.long()
print(y.type())
print(x, y)



x = torch.randn(5, 1)
print(x)
y = torch.gt(x, 1)
print(y.size())

x = np.array([[1], [2]]).reshape(-1)
print(x.shape)
