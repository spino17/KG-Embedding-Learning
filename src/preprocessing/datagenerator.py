from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import torch
from torch.nn.functional import one_hot


class DataGenerator(Dataset):
    """
    class for supporting data preprocessing specific to
    knowledge graph dataset

    """

    def __init__(self, X, Y, Z, target, batch_size, validation_split=0.2):
        super(DataGenerator, self).__init__()
        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)
        Z = torch.from_numpy(Z)
        self.dataset = TensorDataset(X, Y, Z, target)
        self.batch_size = batch_size
        lengths = [
            int(len(self.dataset) * (1 - validation_split)),
            int(len(self.dataset) * validation_split),
        ]
        self.train_dataset, self.validation_dataset = random_split(
            self.dataset, lengths
        )

    def get_trainloader(self):
        return DataLoader(self.train_dataset, self.batch_size)

    def get_validationloader(self):
        return DataLoader(self.validation_dataset, self.batch_size)

    def one_hot_encoding(self, sample, num_entries):
        a = sample[0]  # entity - 1
        b = sample[1]  # entity - 2
        r = sample[2]  # entity - 3
        a = one_hot(sample[0], num_entries).view(-1, num_entries)
        b = one_hot(sample[1], num_entries).view(-1, num_entries)
        r = one_hot(sample[2], num_entries).view(-1, num_entries)
        return a, b, r
