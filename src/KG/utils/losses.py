import torch
from torch import nn


class Losses:
    """
    class for implementing loss functions
    specific to embedding architechures

    """

    def loss_function(name):
        if name == "LogisticLoss":

            def logistic_loss(y_pred, y_target):
                loss = torch.mean(torch.log(1 + torch.exp(-1 * y_pred * y_target)))
                return loss

            return logistic_loss
        elif name == "cross_entropy":
            return nn.CrossEntropyLoss()
        elif name == "NLLLoss":
            return nn.NLLLoss()
