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
        self.embedding_entity = nn.Linear(num_entities, num_dim, bias=False)
        self.embedding_relation = nn.Linear(num_relations, num_dim, bias=False)
        self.sigmoid = nn.Sigmoid()

    # returns the probability for a relation to hold true
    def forward(self, x, y, r):
        embedding_a = self.embedding_entity(x)
        embedding_b = self.embedding_entity(y)
        result_1 = (embedding_a, embedding_b)
        result_2 = torch.mm(self.embedding_relation(r), torch.transpose(result_1, 0, 1))
        result = torch.diag(result_2).view(-1, 1)
        return self.sigmoid(result)
