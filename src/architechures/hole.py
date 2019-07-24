from torch import nn
import torch
import numpy as np


class HOLE(nn.Module):
    """
    Architechure for holographic embeddings - uses circular correlations
    for composition of entities and combining expressive power
    of the tensor product with the efficiency and simplicity of
    TRANSE

    """

    # defining architechure for holographic embedding
    def __init__(self, num_dim, num_entities, num_relations):
        super(HOLE, self).__init__()
        self.num_dim = num_dim
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_list_entity = nn.ModuleList(
            [nn.Linear(1, num_dim) for i in range(num_entities)]
        )
        self.embedding_list_relations = nn.ModuleList(
            [nn.Linear(1, num_dim) for i in range(num_relations)]
        )
        self.sigmoid = nn.Sigmoid()
        self.input_vec = torch.ones(1, 1)  # to convert linear to tensors

    def shift_left(self, a, shift_size):
        a = a.numpy()
        a = np.roll(a, shift_size, axis=1)
        a = torch.from_numpy(a)
        return a

    def circular_correlation(self, a, b):
        result = torch.zeros_like(a)
        for i in range(self.num_dim):
            result[0][i] = torch.mean(a * self.shift_left(b, i)).item()
        return result

    def forward(self, x, y, r):
        result_1 = self.circular_correlation(
            self.embedding_list_entity[x](self.input_vec),
            self.embedding_list_entity[y](self.input_vec),
        )
        result_2 = torch.dot(self.embedding_list_relations[r](self.input_vec), result_1)
        return self.sigmoid(result_2)
