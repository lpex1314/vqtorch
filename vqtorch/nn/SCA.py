import torch
import torch.nn as nn


class SoftClustering(nn.Module):
    def __init__(self, learnable_delta=True, delta=0.2, delta_decay=0.0):
        """
        Initialize the SoftClustering class.

        Args:
            learnable_delta (bool): Whether to use a learnable delta.
            delta (float): The update rate (a constant).
            delta_decay (float): The decay rate of delta (applicable only if delta is learnable).

        """
        super().__init__()

        self.learnable_delta = learnable_delta
        if learnable_delta:
            self.delta = torch.nn.parameter.Parameter(torch.tensor(0.01))
        else:
            self.delta = torch.tensor(delta)
            self.delta_decay = delta_decay

        self.epsilon = 0.01
        # what is a proper epsilon?
        self.sim_matrix = None

    def _compute_similarity(self, v1, v2):
        distances = torch.norm(v1 - v2, p=2, dim=2)
        # similarity = torch.exp(1.0 / (distances + self.epsilon))
        similarity = torch.exp(-distances)
        return similarity

    def update_delta(self):
        self.delta -= self.delta * self.delta_decay

    def update_sim_matrix(self, codebook):
        num_codes, feature_dim = codebook.shape
        codebook_expanded = codebook.unsqueeze(0).expand(num_codes, -1, -1)
        codebook_transposed = codebook_expanded.permute(1, 0, 2)
        self.sim_matrix = self._compute_similarity(
            codebook_expanded, codebook_transposed
        )

    def compute_weighted_sum(self, codebook):
        # compute sim_sum, for computing weight matrix
        sim_sum = torch.sum(self.sim_matrix, dim=1)
        # expand dim of sim_sum
        sim_sum = sim_sum.unsqueeze(1)
        # compute weight matrix
        weights = self.sim_matrix / sim_sum
        # compute weighted_codebook_sum with matrix multiply
        weighted_codebook_sum = torch.matmul(weights, codebook)
        return weighted_codebook_sum

    def forward(self, codebook):
        self.update_sim_matrix(codebook)
        weighted_sum = self.compute_weighted_sum(codebook)
        updated_codebook = ((1 - self.delta) * codebook) + (self.delta * weighted_sum)
        if not self.learnable_delta:
            self.update_delta()
        return updated_codebook
