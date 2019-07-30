from torch import nn
import torch
import numpy as np


class HOLE(nn.Module):
    """
    Architechure for holographic embeddings - uses circular correlations
    for composition of entities and combining expressive power
    of the tensor product with the efficiency and simplicity of
    TRANSE.
    paper - Nickel et al.2015

    """

    # defining architechure for holographic embedding
    def __init__(self, num_dim, num_entities, num_relations):
        super(HOLE, self).__init__()
        self.num_dim = num_dim
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.entity_embedding = nn.Linear(num_entities, num_dim, bias=False)
        self.relation_embedding = nn.Linear(num_relations, num_dim, bias=False)
        self.sigmoid = nn.Sigmoid()

    def shift_left(self, a, shift_size):
        a = a.detach().numpy()
        a = np.roll(a, shift_size, axis=1)
        a = torch.from_numpy(a)
        a = a.clone().detach().requires_grad_(True)
        return a

    def circular_correlation(self, a, b):
        result = torch.mean(a * b, dim=1).view(-1, 1)
        for i in range(1, self.num_dim):
            result = torch.cat(
                [result, torch.mean(a * self.shift_left(b, i), dim=1).view(-1, 1)],
                dim=1,
            )
        return result

    def forward(self, x, y, r):
        result_1 = self.circular_correlation(
            self.entity_embedding(x), self.entity_embedding(y)
        )
        result_2 = torch.mm(self.relation_embedding(r), torch.transpose(result_1, 0, 1))
        result = torch.diag(result_2).view(-1, 1)
        return self.sigmoid(result)
