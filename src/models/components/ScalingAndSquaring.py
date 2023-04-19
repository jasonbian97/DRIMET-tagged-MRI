import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from .SpatialWarp import SpatialWarp

import torch.nn.functional as nnf

class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, nsteps = 7, dim = 3):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialWarp(image_type="image", dim=dim)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec

    def scaling(self,vec):
        return vec * self.scale

    def squaring(self,vec):
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec