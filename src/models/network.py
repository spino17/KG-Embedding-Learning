from torch import nn
import torch
from utils import Losses as L
from utils import Optimizers as O
from utils import Regularizer as R
from preprocessing import DataGenerator


class Network(nn.Module):
    """
    class for forward and backprop for architechures

    """

    def __init__(self, model):
        super(Network, self).__init__()
        self.model = model

    # bind the optimizer and loss function with the model
    def compile(
        self,
        optimizer_name="adam",
        loss="LogisticLoss",
        regularizer="L2",
        lr=0.003,
        momentum=0.95,
        alpha=0.5,
    ):
        self.criterion = L.loss_function(loss)
        self.optimizer = O.optimizer(
            self.model.parameters(), lr, momentum, optimizer_name
        )
        self.regularizer = R.regularizer_function(
            regularizer, self.model.parameters(), alpha
        )

    # training method
    def fit(self, x_train, y_train, batch_size, num_epochs, validation_split=0.2):
        train_losses, test_losses = [], []
        data_processor = DataGenerator(x_train, y_train, batch_size, validation_split)
        TrainLoader = data_processor.get_trainloader()
        ValLoader = data_processor.get_validationloader()
        for epoch in range(num_epochs):
            # training loop
            running_loss = 0.0
            for batch_ndx, sample in enumerate(TrainLoader):
                self.optimizer.zero_grad()
                y_pred = self.model.forward(a, b, r)
                loss = self.criterion(y_pred, y_target) + self.regularizer
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            else:
                # validation loop
                val_loss = 0
                accuracy = 0
                with torch.no_grad():
                    for batch_ndx, sample in enumerate(ValLoader):
                        y_pred = self.model.forward(a, b, r)
                        val_loss += (
                            self.criterion(y_pred, y_target) + self.regularizer
                        ).item()
                        accuracy = (y_pred, y_target)
                train_losses.append(running_loss / len(TrainLoader))
                test_losses.append(val_loss / len(ValLoader))

    # predict the probabilities after training
    def predict(self, a, b, r):
        return self.model.forward(a, b, r)
