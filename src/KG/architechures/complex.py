from torch import nn
import torch


class ComplEx(nn.Module):
    """
    Architechure for modelling anti-symmetric relations (which in real space
    settings can overflow parameters and poses difficulties in generalizations)
    using complex embedding vectors for entities and relations and using hermi-
    tian dot product which stores information of anti-symmetric relations

    """

    def __init__(self, num_dim, num_entities, num_relations):
        super(ComplEx, self).__init__()
        self.num_dim = num_dim
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.entity_embedding_r = nn.Linear(
            num_entities, num_dim, bias=False
        )  # entity embedding real part
        self.entity_embedding_i = nn.Linear(
            num_entities, num_dim, bias=False
        )  # entity embedding imaginary part
        self.relation_embedding_r = nn.Linear(
            num_relations, num_dim, bias=False
        )  # relation embedding real part
        self.relation_embedding_i = nn.Linear(
            num_relations, num_dim, bias=False
        )  # relation embedding imaginary part
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y, r):
        term_1 = torch.mean(
            self.relation_embedding_r(r)
            * self.entity_embedding_r(x)
            * self.entity_embedding_r(y),
            dim=1,
        ).view(-1, 1)
        term_2 = torch.mean(
            self.relation_embedding_r(r)
            * self.entity_embedding_i(x)
            * self.entity_embedding_i(y),
            dim=1,
        ).view(-1, 1)
        term_3 = torch.mean(
            self.relation_embedding_i(r)
            * self.entity_embedding_r(x)
            * self.entity_embedding_i(y),
            dim=1,
        ).view(-1, 1)
        term_4 = torch.mean(
            self.relation_embedding_i(r)
            * self.entity_embedding_i(x)
            * self.entity_embedding_r(y),
            dim=1,
        ).view(-1, 1)
        result = (
            term_1 + term_2 + term_3 - term_4
        )  # resultant tensor product of triplet embeddings (x, y, r)
        return self.sigmoid(result)
