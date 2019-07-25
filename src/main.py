from architechures import Compositional, HOLE
from models import Network
import torch
from torch import nn

# declaring hyperparameters
num_dim = 50  # dimension of embedding vector
num_entities = 100  # number of words in vocabulary
num_relations = 100  # number of relations
alpha = 1  # coefficient of regularization term
batch_size = 64  # backprop for this many combined datapoints
num_epochs = 10  # number of loops over training dataset

# data preprocessing
# TODO

# defining the model
model = Network(Compositional(num_dim, num_entities, num_relations))

# compiling the model
model.compile(optimizer_name="adam", loss="LogisticLoss")

# training call
model.fit(x_train, y_train, batch_size, num_epochs)

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

x = torch.randn((1, 10))
y = torch.randn((1, 10))
z = torch.randn((1, 10))
t = x * y * z
print(t)
print(t.size())
print(x[0][0], y[0][0], z[0][0])
