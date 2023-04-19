import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pystrum.pynd.ndutils as nd

from torch import nn
from functools import partial
from einops import rearrange, repeat


def jacobian_determinant_numpy(disp):
    """
    jacobian determinant of a displacement field.
    Assume no batch dimension.
    NB: to compute the spatial gradients, we use np.gradient.
    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims
    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # 3D glow
    if nb_dims == 3:
        grid = grid[..., [1, 0, 2]]
        # compute gradients
        J = np.gradient(disp + grid)

        dx = J[1]
        dy = J[0]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # must be 2
        grid = grid[..., [1, 0]]
        # compute gradients
        J = np.gradient(disp + grid)

        dfdx = J[1]
        dfdy = J[0]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]

def jacobian_determinant_tensor(disp):
    """
    jacobian determinant of a displacement field.
    Assume batch dimension.
    NB: to compute the spatial gradients, we use np.gradient.
    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims
    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    nb_dims = len(disp.shape)
    assert nb_dims in (4, 5), 'flow has to be 2D or 3D'
    device = disp.device
    # 3D flow
    if nb_dims == 5:
        if disp.shape[1]==3:
            disp = rearrange(disp, 'b c x y z -> b x y z c')
        b, h, w, d, c = disp.shape
        assert c == 3, 'flow has to be 3D and channel-last'
        grid_h, grid_w, grid_d = torch.meshgrid(
            [torch.linspace(0, h - 1, h),
             torch.linspace(0, w - 1, w),
             torch.linspace(0, d - 1, d)])  # compute grid
        grid = torch.stack([grid_w, grid_h, grid_d], dim=3).type_as(disp)

        dx = torch.empty(b, h, w, d, 3).type_as(disp)
        dy = torch.empty(b, h, w, d, 3).type_as(disp)
        dz = torch.empty(b, h, w, d, 3).type_as(disp)

        for bb in range(b):
            trans = disp[bb] + grid
            J = torch.gradient(trans)
            dx[bb] = J[1] # PAY ATTENTION: torch.gradient 's first dimention is in matrix index, which is our y direction
            dy[bb] = J[0]
            dz[bb] = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        Det = Jdet0 - Jdet1 + Jdet2

        # add channel dimension
        Det = rearrange(Det, 'b x y z -> b 1 x y z')
        return Det

    else:  # must be 2D
        if disp.shape[1]==2:
            disp = rearrange(disp, 'b c x y -> b x y c')
        b, h, w, c = disp.shape
        assert c == 2, 'flow has to be 2D and channel-last'
        grid_h, grid_w = torch.meshgrid(
            [torch.linspace(0, h-1, h), torch.linspace(0, w-1, w)]) # compute grid
        grid = torch.stack([grid_w, grid_h], dim=2).to(device)
        dx = torch.empty(b, h, w, 2).to(device)
        dy = torch.empty(b, h, w, 2).to(device)

        for bb in range(b):
            trans = disp[bb] + grid
            J = torch.gradient(trans)
            dx[bb] = J[1]
            dy[bb] = J[0]

        return dx[..., 0] * dy[..., 1] - dy[..., 0] * dx[..., 1]
