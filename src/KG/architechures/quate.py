from torch import nn
import torch
import numpy as np


class QuatE(nn.Module):
    """
    Architechure for hypercomplex embeddings - exploits the homomorphism between
    its rotation symmetry group su(2) with the spacial rotation group so(3) and
    thus have one more rotational degree of freedom than ComplEx representation.
    paper - Zhang et al.2019

    """

    def __init__(self, num_dim, num_entities, num_relations):
        super(QuatE, self).__init__()
        self.num_dim = num_dim
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_entity_r = nn.Linear(
            num_entities, num_dim, bias=False
        )  # entity real part
        self.embedding_entity_i = nn.Linear(
            num_entities, num_dim, bias=False
        )  # entity imaginary i part
        self.embedding_entity_j = nn.Linear(
            num_entities, num_dim, bias=False
        )  # entity imaginary j part
        self.embedding_entity_k = nn.Linear(
            num_entities, num_dim, bias=False
        )  # entity imaginary k part
        self.embedding_relation_r = nn.Linear(
            num_relations, num_dim, bias=False
        )  # relation real part
        self.embedding_relation_i = nn.Linear(
            num_relations, num_dim, bias=False
        )  # relation imaginary i part
        self.embedding_relation_j = nn.Linear(
            num_relations, num_dim, bias=False
        )  # relation imaginart j part
        self.embedding_relation_k = nn.Linear(
            num_relations, num_dim, bias=False
        )  # relation imaginary k part
        self.sigmoid = nn.Sigmoid()

    # calculate hypercomplex norm of a quaternion vector
    def hypercomplex_norm(self, a_r, a_i, a_j, a_k):
        return torch.sqrt(
            torch.norm(a_r, dim=1) ** 2
            + torch.norm(a_i, dim=1) ** 2
            + torch.norm(a_j, dim=1) ** 2
            + torch.norm(a_k, dim=1) ** 2
        ).view(-1, 1)

    # hamilton product as defined in Zhang et al.2019
    def hamilton_product(self, a_r, a_i, a_j, a_k, b_r, b_i, b_j, b_k):
        q_r = a_r * b_r - a_i * b_i - a_j * b_j - a_k * b_k
        q_i = a_r * b_i + a_i * b_r + a_j * b_k - a_k * b_j
        q_j = a_r * b_j - a_i * b_k + a_j * b_r + a_k * b_i
        q_k = a_r * b_k + a_i * b_j - a_j * b_i + a_k * b_r
        return q_r, q_i, q_j, q_k

    def forward(self, h, t, r):
        # quaternion embeddings for the triplet
        H_r = self.embedding_entity_r(h)
        H_i = self.embedding_entity_i(h)
        H_j = self.embedding_entity_j(h)
        H_k = self.embedding_entity_k(h)

        T_r = self.embedding_entity_r(t)
        T_i = self.embedding_entity_i(t)
        T_j = self.embedding_entity_j(t)
        T_k = self.embedding_entity_k(t)

        R_r = self.embedding_relation_r(r)
        R_i = self.embedding_relation_i(r)
        R_j = self.embedding_relation_j(r)
        R_k = self.embedding_relation_k(r)

        norm = self.hypercomplex_norm(R_r, R_i, R_j, R_k)
        R_r = norm * R_r
        R_i = norm * R_i
        R_j = norm * R_j
        R_k = norm * R_k

        """
        Q_r = H_r * R_r - H_i * R_i - H_j * R_j - H_k * R_k
        Q_i = H_r * R_i + H_i * R_r + H_j * R_k - H_k * R_j
        Q_j = H_r * R_j - H_i * R_k + H_j * R_r + H_k * R_i
        Q_k = H_r * R_k + H_i * R_j - H_j * R_i + H_k * R_r
        """
        Q_r, Q_i, Q_j, Q_k = self.hamilton_product(
            H_r, H_i, H_j, H_k, R_r, R_i, R_j, R_k
        )

        term_1 = torch.diag(torch.mm(Q_r, torch.transpose(T_r, 0, 1))).view(-1, 1)
        term_2 = torch.diag(torch.mm(Q_i, torch.transpose(T_i, 0, 1))).view(-1, 1)
        term_3 = torch.diag(torch.mm(Q_j, torch.transpose(T_j, 0, 1))).view(-1, 1)
        term_4 = torch.diag(torch.mm(Q_k, torch.transpose(T_k, 0, 1))).view(-1, 1)

        result = term_1 + term_2 + term_3 + term_4
        return self.sigmoid(result)
