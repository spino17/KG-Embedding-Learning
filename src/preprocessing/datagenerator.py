from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split


class DataGenerator(Dataset):
    """
    class for supporting data preprocessing specific to
    knowledge graph dataset

    """

    def __init__(self, X, Y, batch_size, validation_split=0.2):
        super(DataGenerator, self).__init__()
        self.dataset = TensorDataset(X, Y)
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
