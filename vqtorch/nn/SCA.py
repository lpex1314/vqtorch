import torch


class SoftClustering:
    def __init__(self, delta=0.2, lr=0.0):
        """
        Initialize the SoftClustering class.

        Args:
            delta (float): The update rate (a constant).
        """
        self.delta = delta
        self.epsilon = torch.tensor(0.01)
        # what is a proper epsilon?
        self.lr = lr
        self.sim_matrix = None

    def _compute_similarity(self, v1, v2):
        distances = torch.norm(v1 - v2, p=2, dim=2)
        similarity = torch.exp(1.0 / (distances + self.epsilon))
        return similarity

    def update_delta(self):
        self.delta -= self.delta * self.lr

    # @torch.no_grad()
    # def update_sim_matrix(self, codebook):
    #     """
    #     Perform soft clustering update on signal points.
    #
    #     Args:
    #         codebook (torch.Tensor): The original codebook.
    #
    #     Returns:
    #         newcodebook (torch.Tensor): The updated codebook.
    #     """
    #
    #     num_codes, feature_dim = codebook.shape
    #
    #     codebook_expanded = codebook.unsqueeze(0).expand(num_codes, -1, -1)
    #     codebook_transposed = codebook_expanded.permute(1, 0, 2)
    #     distances = torch.norm(codebook_expanded - codebook_transposed, p=2, dim=2)
    #     self.sim_matrix = torch.exp(1.0 / (distances + self.epsilon))
    #
    #     del distances, codebook_expanded, codebook_transposed
    #     # return codebook
    @torch.no_grad()
    def update_sim_matrix(self, codebook):
        """
        Perform soft clustering sim_matrix update with signal points.

        Args:
            codebook (torch.Tensor): The original codebook.

        Returns:
            new sim_matrix (torch.Tensor): The updated sim_matrix.
        """

        num_codes, feature_dim = codebook.shape

        codebook_expanded = codebook.unsqueeze(0).expand(num_codes, -1, -1)
        codebook_transposed = codebook_expanded.permute(1, 0, 2)
        distances = torch.norm(codebook_expanded - codebook_transposed, p=2, dim=2)
        self.sim_matrix = torch.exp(1.0 / (distances + self.epsilon))

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
