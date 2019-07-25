from torch import nn
import torch


class Compositional(nn.Module):
    """
    Architechure for modelling symmetric relations using
    compositional (dot product) embeddings

    """

    # defining architechure given in paper Nickel et al (2015) -
    # compositional embeddings of knowledge graph
    def __init__(self, num_dim, num_entities, num_relations):
        super(Compositional, self).__init__()
        # index in module list is equal to the index of the object
        self.num_dim = num_dim
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_list_entity = nn.ModuleList(
            [nn.Linear(1, num_dim) for i in range(num_entities)]
        )
        self.embedding_list_relations = nn.ModuleList(
            [nn.Linear(num_dim, num_dim) for i in range(num_relations)]
        )
        self.sigmoid = nn.Sigmoid()
        self.input_vec_1 = torch.ones(1, 1)  # to convert linear to tensors
        self.input_vec_2 = torch.eye(num_dim)  # identity matrix

    # returns the probability for a relation to hold true
    def forward(self, x, y, r):
        result_1 = torch.mm(
            self.embedding_list_entity[y](self.input_vec_1),
            self.embedding_list_relations[r](self.input_vec_2),
        )
        result_2 = torch.dot(result_1, self.embedding_list_entity[x](self.input_vec_1))
        return self.sigmoid(result_2)
