from KG.architechures import Compositional, HOLE, ComplEx, QuatE
from KG.models import Network
from torch import nn
import numpy as np
from KG.preprocessing import DataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch


# data preprocessing
obj = DataGenerator()
X, y, num_entities, num_relations = obj.load_dataset(
    "/home/bhavya/Desktop/projects/KG-Embedding-Learning---PyTorch/dataset/siemens data/01.nt"
)

# split dataset randomly in training and testing
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=0
)

# declaring hyperparameters
num_dim = 10  # dimension of embedding vector
alpha = 0  # coefficient of regularization term in total loss function
batch_size = 20  # backprop for this many combined datapoints
num_epochs = 30  # number of loops over training dataset

# defining the model
model = Network(
    QuatE(num_dim, num_entities, num_relations)
)  # using QuatE architechure for modelling
model.compile(optimizer_name="adam", loss="LogisticLoss", regularizer="L2", alpha=alpha)

# training the model
model.fit(x_train, y_train, batch_size, num_epochs, validation_split=0.2)


# performance evaluation and predictions on testing dataset
y_pred = torch.gt(model.predict(x_test), 0.5).long().numpy()
accuracy = accuracy_score(y_test, y_pred)
print("accuracy on testing set: ", accuracy * 100)
