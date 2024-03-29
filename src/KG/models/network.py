from torch import nn
import torch
from KG.utils import Losses as L
from KG.utils import Optimizers as O
from KG.utils import Regularizer as R
from KG.preprocessing import DataGenerator
from torch.utils.data import TensorDataset, Dataset, DataLoader
from torch.nn.functional import one_hot
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


class Network(nn.Module):
    """
    class for forward and backprop functionalities for architechures

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
        alpha=0,
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
        train_losses, val_losses, epochs = [], [], []
        X = x_train[:, 0]  # entity - 1
        Y = x_train[:, 1]  # entity - 2
        Z = x_train[:, 2]  # entity - 3
        y_target = y_train  # target probabilities
        data_processor = DataGenerator()
        data_processor.set_dataset(
            X, Y, Z, y_target, batch_size, validation_split
        )  # tensorise the dataset elements for further processing in pytorch nn module
        TrainLoader = data_processor.get_trainloader()
        ValLoader = data_processor.get_validationloader()
        for epoch in range(num_epochs):
            print("epoch no. ", epoch + 1)
            # training loop
            train_loss = 0
            self.model.train()
            for batch_ndx, sample in enumerate(TrainLoader):
                # print("---batch no. ", batch_ndx + 1)
                a = data_processor.one_hot_encoding(sample[0], self.model.num_entities)
                b = data_processor.one_hot_encoding(sample[1], self.model.num_entities)
                r = data_processor.one_hot_encoding(sample[2], self.model.num_relations)
                y_target = sample[3]  # target probabilities
                self.optimizer.zero_grad()
                y_pred = self.model(a, b, r)
                # print(y_pred)
                loss = self.criterion(y_pred, y_target) + self.regularizer
                loss.backward(retain_graph=True)
                self.optimizer.step()
                train_loss += loss.item()
            else:
                # validation loop
                self.model.eval()
                val_loss = 0
                accuracy = 0
                with torch.no_grad():
                    # scope of no gradient calculations
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
                        y_hat = torch.gt(y_pred, 0.5).long().numpy().reshape(-1)
                        y_true = y_target.long().numpy().reshape(-1)
                        # accuracy = accuracy_score(y_true, y_hat)
                        # print("accuracy on validation set: ", accuracy * 100)
                        accuracy += accuracy_score(y_true, y_hat)
                print("accuracy on validation set: ", accuracy / len(ValLoader))
                train_losses.append(train_loss / len(TrainLoader))
                val_losses.append(val_loss / len(ValLoader))
                epochs.append(epoch + 1)

        # plot the loss vs epoch graphs
        plt.plot(epochs, train_losses, color="red")
        plt.plot(epochs, val_losses, color="blue")
        plt.show()

    # predict the probabilities after training
    def predict(self, x_test):
        batch_size = int(x_test.size / 3)  # number of rows in test dataset
        X = torch.from_numpy(x_test[:, 0])  # entity - 1
        Y = torch.from_numpy(x_test[:, 1])  # entity - 2
        Z = torch.from_numpy(x_test[:, 2])  # entity - 3
        test_dataset = TensorDataset(X, Y, Z)
        TestLoader = DataLoader(test_dataset, batch_size)
        with torch.no_grad():
            # scope of no gradient calculations
            self.model.eval()
            for batch_ndx, sample in enumerate(TestLoader):
                a = one_hot(sample[0], self.model.num_entities).float()
                b = one_hot(sample[1], self.model.num_entities).float()
                r = one_hot(sample[2], self.model.num_relations).float()
                break
            return self.model.forward(a, b, r)
