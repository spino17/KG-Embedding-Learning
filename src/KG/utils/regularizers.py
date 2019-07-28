import torch
from torch import nn


class Regularizer:
    """
    class for implementing regularizer

    """

    def regularizer_function(name, parameters, alpha=0.5):
        if name == "L2":
            L2_term = 0
            for params in parameters:
                L2_term += torch.norm(params) ** 2
            return alpha * L2_term
        elif name == "L1":
            L1_term = 0
            for params in parameters:
                L1_term += torch.norm(params)
            return alpha * L1_term
