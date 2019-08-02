from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import torch
from torch.nn.functional import one_hot
import numpy as np
from rdflib.graph import Graph
import json
from sklearn.utils import shuffle
import random


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

        # initializing adjacency matrix - 3D matrix which has first, second dimension
        # for entity indices and thrid dimension for relation indices and is equal
        # to one when the triplet fact is true else false

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
        num_entities = len(triplet_dict_entity)
        num_relations = len(triplet_dict_relation)
        labels = np.ones((len(dataset), 1))  # labels

        # initializing adjacency matrix - 3D matrix which has first, second dimension
        # for entity indices and thrid dimension for relation indices and is equal
        # to one when the triplet fact is true else false
        self.adj_matrix = np.zeros((num_entities, num_entities, num_relations))
        for row in dataset:
            idx_0 = int(row[0])  # entity s mapped integer
            idx_1 = int(row[1])  # entity o mapped integer
            idx_2 = int(row[2])  # relation mapped integer
            self.adj_matrix[idx_0][idx_1][idx_2] = 1  # make the triplet entry 1

        # randomly synthesing false facts
        critical_size = len(
            dataset
        )  # false facts should be upto this critical size which is size of true fact triplet dataset
        count = 0
        while count < critical_size:
            idx_0 = random.randrange(0, num_entities - 1)
            idx_1 = random.randrange(0, num_entities - 1)
            idx_2 = random.randrange(0, num_relations - 1)
            if self.adj_matrix[idx_0][idx_1][idx_2] == 0:
                count += 1
                dataset.append([idx_0, idx_1, idx_2])

        X = np.array(dataset)  # triplets of true and false facts
        y = np.concatenate((labels, np.zeros((count, 1))), axis=0)
        X, y = shuffle(X, y)  # randomly shuffle the dataset

        return X, y, num_entities, num_relations
