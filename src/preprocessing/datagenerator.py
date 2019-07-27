from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import torch
from torch.nn.functional import one_hot


class DataGenerator(Dataset):
    """
    class for supporting data preprocessing specific to
    knowledge graph dataset

    """

    def __init__(self, X, Y, Z, y_target, batch_size, validation_split=0.2):
        super(DataGenerator, self).__init__()
        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)
        Z = torch.from_numpy(Z)
        y_target = torch.from_numpy(y_target)
        self.dataset = TensorDataset(X, Y, Z, y_target)
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
        sample = one_hot(sample, num_entries).view(-1, num_entries)
        return sample
