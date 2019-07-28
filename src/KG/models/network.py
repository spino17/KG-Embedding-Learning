from torch import nn
import torch
from KG.utils import Losses as L
from KG.utils import Optimizers as O
from KG.utils import Regularizer as R
from KG.preprocessing import DataGenerator
from torch.utils.data import TensorDataset, Dataset, DataLoader


class Network(nn.Module):
    """
    class for forward and backprop for architechures

    """

    def __init__(self, model):
        super().__init__()
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
        X = x_train[:, 0]  # entity - 1
        Y = x_train[:, 1]  # entity - 2
        Z = x_train[:, 2]  # entity - 3
        y_target = y_train  # target probabilities
        data_processor = DataGenerator(X, Y, Z, y_target, batch_size, validation_split)
        TrainLoader = data_processor.get_trainloader()
        ValLoader = data_processor.get_validationloader()
        for epoch in range(num_epochs):
            print("epoch no. ", epoch + 1)
            # training loop
            running_loss = 0.0
            for batch_ndx, sample in enumerate(TrainLoader):
                print("-----batch no. ", batch_ndx + 1)
                a = data_processor.one_hot_encoding(sample[0], self.model.num_entities)
                b = data_processor.one_hot_encoding(sample[1], self.model.num_entities)
                r = data_processor.one_hot_encoding(sample[2], self.model.num_relations)
                y_target = sample[3]  # target probabilities
                self.optimizer.zero_grad()
                y_pred = self.model(a, b, r)
                loss_1 = self.criterion(y_pred, y_target)
                #loss_2 = self.regularizer
                print(loss_1)
                #loss.backward(retain_graph=True)
                loss_1.backward()
                #loss_2.backward(retain_graph=True)
                self.optimizer.step()
                #running_loss += (loss_1 + loss_2).item()
            """
            else:
                # validation loop
                val_loss = 0
                accuracy = 0
                with torch.no_grad():
                    for batch_ndx, sample in enumerate(ValLoader):
                        a = data_processor.one_hot_encoding(
                            sample[0], self.model.num_entities
                        )
                        b = data_processor.one_hot_encoding(
                            sample[1], self.model.num_entities
                        )
                        r = data_processor.one_hot_encoding(
                            sample[2], self.model.num_relations
                        )
                        y_target = sample[3]
                        y_pred = self.model(a, b, r)
                        val_loss += (
                            self.criterion(y_pred, y_target) + self.regularizer
                        ).item()
                        accuracy = (y_pred, y_target)
                train_losses.append(running_loss / len(TrainLoader))
                test_losses.append(val_loss / len(ValLoader))
            """

    # predict the probabilities after training
    def predict(self, x_test):
        X = x_test[:, 0]  # entity - 1
        Y = x_test[:, 1]  # entity - 2
        Z = x_test[:, 2]  # entity - 3
        test_dataset = TensorDataset(X, Y, Z)
        return self.model.forward(a, b, r)
