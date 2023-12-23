import torch
import torch.nn as nn
import torch.nn.functional as F

from vqtorch.dists import get_dist_fns
import vqtorch
from vqtorch.norms import with_codebook_normalization
from .vq_base import _VQBaseLayer
from .affine import AffineTransform
from .SCA import SoftClustering


class SoftVectorQuant(_VQBaseLayer):
    """
    Vector quantization layer using straight-through estimation.

    Args:
            feature_size (int): feature dimension corresponding to the vectors
            num_codes (int): number of vectors in the codebook
            beta (float): commitment loss weighting
            use_sd (bool): use soft discretization
            gamma (float): gamma for soft discretization
            use_sca (bool): use soft clustering assignment
            learnable_delta (bool): whether delta is learnable for soft clustering assignment
            init_delta (float): initial delta value for soft clustering assignment
            delta_decay (float): delta decay rate for soft clustering assignment
            inplace_optimizer (Optimizer): optimizer for inplace codebook updates
            **kwargs: additional arguments for _VQBaseLayer

    Returns:
            Quantized vector z_q and return dict
    """

    def __init__(
        self,
        feature_size: int,
        num_codes: int,
        beta: float = 0.95,
        use_sd: bool = True,
        gamma: float = 0.0,
        use_sca=True,
        learnable_delta=True,
        init_delta: float = 0.0,
        delta_decay: float = 0.0,
        inplace_optimizer: torch.optim.Optimizer = None,
        **kwargs,
    ):
        super().__init__(feature_size, num_codes, **kwargs)
        self.loss_fn, self.dist_fn = get_dist_fns("euclidean")

        if beta < 0.0 or beta > 1.0:
            raise ValueError(f"beta must be in [0, 1] but got {beta}")

        if init_delta < 0.0 or init_delta > 1.0:
            raise ValueError(f"delta must be in [0, 1] but got {init_delta}")

        self.beta = beta
        self.codebook = nn.Embedding(self.num_codes, self.feature_size)
        self.use_sd = use_sd
        self.use_sca = use_sca

        if self.use_sd:
            self.gamma = gamma

        if inplace_optimizer is not None:
            if beta != 1.0:
                raise ValueError("inplace_optimizer can only be used with beta=1.0")
            self.inplace_codebook_optimizer = inplace_optimizer(
                self.codebook.parameters()
            )

        if self.use_sca:
            self.sca = SoftClustering(
                learnable_delta=learnable_delta,
                delta=init_delta,
                delta_decay=delta_decay,
            )

        return

    def straight_through_approximation(self, z, z_q):
        """passed gradient from z_q to z"""
        z_q = z + (z_q - z).detach()
        return z_q

    def compute_loss(self, z_e, z_q):
        """computes loss between z and z_q"""
        return (1.0 - self.beta) * self.loss_fn(z_e, z_q.detach()) + (
            self.beta
        ) * self.loss_fn(z_e.detach(), z_q)

    def quantize(self, codebook, z):
        """
        Quantizes the latent codes z with the codebook

        Args:
                codebook (Tensor): B x F
                z (Tensor): B x ... x F
        """

        # reshape to (BHWG x F//G) and compute distance
        z_shape = z.shape[:-1]
        z_flat = z.view(z.size(0), -1, z.size(-1))

        print("yoyo", codebook.requires_grad)

        if self.use_sca:
            codebook = self.sca(codebook)
            print(codebook.requires_grad)

        with torch.no_grad():
            dist_out = self.dist_fn(
                tensor=z_flat,
                codebook=codebook,
                topk=self.topk,
                compute_chunk_size=self.cdist_chunk_size,
                half_precision=(z.is_cuda),
            )

            d = dist_out["d"].view(z_shape)
            q = dist_out["q"].view(z_shape).long()

        z_q = F.embedding(q, codebook)

        if self.use_sd:
            z_q = (1 - self.gamma) * z_q + self.gamma * z_flat

        if self.training and hasattr(self, "inplace_codebook_optimizer"):
            # update codebook inplace
            ((z_q - z.detach()) ** 2).mean().backward()
            self.inplace_codebook_optimizer.step()
            self.inplace_codebook_optimizer.zero_grad()

            # forward pass again with the update codebook
            z_q = F.embedding(q, codebook)

            # NOTE to save compute, we assumed Q did not change.

        return z_q, d, q

    @torch.no_grad()
    def get_codebook(self):
        cb = self.codebook.weight
        print(cb.requires_grad)
        if hasattr(self, "sca"):
            cb = self.sca(cb)
            print(cb.requires_grad)
        return cb

    def get_delta(self):
        if hasattr(self, "sca"):
            return self.sca.delta.item()
        return None

    @with_codebook_normalization
    def forward(self, z):
        ######
        ## (1) formatting data by groups and invariant to dim
        ######

        z = self.prepare_inputs(z, self.groups)

        if not self.enabled:
            z = self.to_original_format(z)
            return z, {}

        ######
        ## (2) quantize latent vector
        ######

        z_q, d, q = self.quantize(self.codebook.weight, z)

        # e_mean = F.one_hot(q, num_classes=self.num_codes).view(-1, self.num_codes).float().mean(0)
        # perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        perplexity = None

        to_return = {
            "z": z,  # each group input z_e
            "z_q": z_q,  # quantized output z_q
            "d": d,  # distance function for each group
            "q": q,  # codes
            "loss": self.compute_loss(z, z_q).mean(),
            "perplexity": perplexity,
        }

        z_q = self.straight_through_approximation(z, z_q)
        z_q = self.to_original_format(z_q)

        return z_q, to_return
