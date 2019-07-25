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
        self.embedding_entity_r = nn.ModuleList(
            [nn.Linear(1, num_dim) for i in range(num_entities)]
        )  # entity embedding real part
        self.embedding_entity_i = nn.ModuleList(
            [nn.Linear(1, num_dim) for i in range(num_entities)]
        )  # entity embedding imaginary part
        self.embedding_relation_r = nn.ModuleList(
            [nn.Linear(1, num_dim) for i in range(num_entities)]
        )  # relation embedding real part
        self.embedding_relation_i = nn.ModuleList(
            [nn.Linear(1, num_dim) for i in range(num_entities)]
        )  # relation embedding imaginary part
        self.input_vec = torch.ones(1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y, r):
        term_1 = torch.mean(
            self.embedding_relation_r[r](self.input_vec)
            * self.embedding_entity_r[x](self.input_vec)
            * self.embedding_entity_r[y](self.input_vec)
        )
        term_2 = torch.mean(
            self.embedding_relation_r[r](self.input_vec)
            * self.embedding_entity_i[x](self.input_vec)
            * self.embedding_entity_i[y](self.input_vec)
        )
        term_3 = torch.mean(
            self.embedding_relation_i[r](self.input_vec)
            * self.embedding_entity_r[x](self.input_vec)
            * self.embedding_entity_i[y](self.input_vec)
        )
        term_4 = torch.mean(
            self.embedding_relation_i[r](self.input_vec)
            * self.embedding_entity_i[x](self.input_vec)
            * self.embedding_entity_r[y](self.input_vec)
        )
        result = (
            term_1 + term_2 + term_3 - term_4
        )  # resultant tensor product of triplet embeddings (x, y, r)
        return self.sigmoid(result)
