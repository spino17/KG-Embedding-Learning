import torch.optim as optim


class Optimizers:
    """
    class for built-in optimizers

    """

    def optimizer(params, learning_rate, momentum, name):
        if name == "SGD":
            return optim.SGD(params, learning_rate, momentum, nesterov=False)
        elif name == "adam":
            return optim.Adam(params, learning_rate)
