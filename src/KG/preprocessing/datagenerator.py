from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import torch
from torch.nn.functional import one_hot
import numpy as np
from rdflib.graph import Graph
import json


class DataGenerator(Dataset):
    """
    class for supporting data preprocessing specific to
    knowledge graph dataset

    """

    def __init__(self):
        super().__init__()

    def set_dataset(self, X, Y, Z, y_target, batch_size, validation_split=0.2):
        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)
        Z = torch.from_numpy(Z)
        y_target = torch.from_numpy(y_target).float()
        self.dataset = TensorDataset(X, Y, Z, y_target)
        self.batch_size = batch_size
        training_length = int(len(self.dataset) * (1 - validation_split))
        lengths = [training_length, len(self.dataset) - training_length]
        self.train_dataset, self.validation_dataset = random_split(
            self.dataset, lengths
        )

    def get_trainloader(self):
        return DataLoader(self.train_dataset, self.batch_size)

    def get_validationloader(self):
        return DataLoader(self.validation_dataset, self.batch_size)

    def one_hot_encoding(self, sample, num_entries):
        sample = one_hot(sample, num_entries).view(-1, num_entries)
        return sample.float()

    def load_dataset(self, file_path):
        g = Graph()
        g.parse(
            file_path, format="nt"
        )  # file to parse data from (should be in nt format)

        # loop in g for each triplet
        index_e, index_r = 0, 0  # each entity and relation is mapped to a integer
        triplet_dict_entity = {}  # mapping dictionary from entities to integers
        triplet_dict_relation = {}  # mapping dictionary from relations to integers
        counter_e = []
        counter_r = []
        dataset = []
        for stmt in g:
            subject = stmt[0]  # subject entity
            relation = stmt[1]  # predicate relation
            object = stmt[2]  # object entity
            if subject in triplet_dict_entity.keys():
                index = triplet_dict_entity[subject]
                counter_e[index] += 1
            else:
                triplet_dict_entity[subject] = index_e
                counter_e.append(1)
                index_e += 1

            if object in triplet_dict_entity.keys():
                index = triplet_dict_entity[object]
                counter_e[index] += 1
            else:
                triplet_dict_entity[object] = index_e
                counter_e.append(1)
                index_e += 1

            if relation in triplet_dict_relation.keys():
                index = triplet_dict_relation[relation]
                counter_r[index] += 1
            else:
                triplet_dict_relation[relation] = index_r
                counter_r.append(1)
                index_r += 1

            int_s = triplet_dict_entity[subject]
            int_o = triplet_dict_entity[object]
            int_r = triplet_dict_relation[relation]
            dataset.append([int_s, int_o, int_r])

        self.entity_to_integer_mapping = triplet_dict_entity
        self.relation_to_integer_mapping = triplet_dict_relation
        return np.array(dataset), len(triplet_dict_entity), len(triplet_dict_relation)
