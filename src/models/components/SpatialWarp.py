"""Implemented by Zhangxing Bian 2022/11"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pystrum.pynd.ndutils as nd

from torch import nn
from functools import partial
from einops import rearrange, repeat
from typing import Optional

# def wrap_2pi(x):
#     return torch.remainder(x + np.pi, 2 * np.pi) - np.pi

def wrap_any(x, mod):
    # return torch.fmod(x + np.pi, 2 * np.pi) - np.pi # does not work
    # Note: the input x needs to be zero-centered, and max-min = mod
    return torch.remainder(x + mod/2, mod) - mod/2


def scale_flow(flow, factor: tuple):
    # factor is a tuple of 3, represents multiplicative factor to x, y, z
    if len(flow.shape) == 5:
        flow[:,0,...] = flow[:,0,...] * factor[0] # assume flow is channel first and has batch dimentions
        flow[:,1,...] = flow[:,1,...] * factor[1]
        flow[:,2,...] = flow[:,2,...] * factor[2]
    elif len(flow.shape) == 4: # 2D flow
        flow[:,0,...] = flow[:,0,...] * factor[0]
        flow[:,1,...] = flow[:,1,...] * factor[1]
    else:
        raise ValueError("Unsupported flow shape: {}".format(flow.shape))
    return flow

class SpatialWarp(nn.Module):
    """
    """
    def __init__(
        self,
        image_type: str = "image", # can be "phase" or "image"
        dim: int = 3, # can be 2 or 3
        mod: Optional[float] = 1.0,
    ):
        super().__init__()
        self.image_type = image_type
        self.dim = dim
        self.mod = mod

        if image_type == "phase":
            self.wrap = partial(wrap_any, mod=mod)

        if image_type == "image":
            if self.dim == 2:
                self.warp = flow_warp
            else: # dim==3
                self.warp = flow_warp_3d
        elif image_type == "phase":
            if self.dim == 2:
                self.warp = partial(self.flow_sampler_2d_in_batch, isPhase=True)
            else: # dim==3
                self.warp = partial(self.flow_sampler_3d_in_batch, isPhase=True)
        else:
            raise ValueError("Unsupported image type and dim combination. Choose from: {image, phase}, {2, 3}")

    def forward(self, x, disp):
        if self.image_type == "image":
            # print("maximum disp:", torch.max(disp).item(), "min disp:", torch.min(disp).item(), "  mean disp:", torch.mean(disp).item())
            return self.warp(x, disp)
        elif self.image_type == "phase":
            b,c,h,w,d = disp.shape
            x = x - self.mod/2. # Since input is not zero-centered, then we need to shift it to zero-centered
            disp = scale_flow(disp, factor = (w,h,d)) # scale up the flow to the original image size
            warped = self.warp(x, disp) + self.mod/2. # shift back to the original range
            # print("maximum disp:", torch.max(disp).item(), "min disp:", torch.min(disp).item(), "  mean disp:", torch.mean(disp).item())
            return warped

    def flow_sampler_3d_in_batch(self, img, flow, isPhase=False):
        """flow: B, H, W, D, 3
        img: B, C, H, W, D"""
        if flow.shape[1] == 3:
            flow = rearrange(flow, 'b c h w d -> b h w d c')
        sample_grid = flow_to_grid_3d(flow)
        warped = self.grid_sampler_3d_in_batch(img, sample_grid, isPhase=isPhase)
        return warped

    def flow_sampler_2d_in_batch(self, img, flow, isPhase=False):
        """flow: B, H, W, 2
        img: B, C, H, W"""
        sample_grid = flow_to_grid_2d(flow)
        warped = self.grid_sampler_2d_in_batch(img, sample_grid, isPhase=isPhase)
        return warped

    def linear_interpolation_1d(self, x1, x2, fx1, fx2, x, isPhase=False):
        if isPhase:
            return fx1 + self.wrap(self.wrap(fx2 - fx1) / (x2 - x1) * (x - x1))
        else:
            return fx1 + (fx2 - fx1) / (x2 - x1) * (x - x1)

    def bilinear_interpolation_2d(self, x1, x2, y1, y2, fx11, fx12, fx21, fx22, x, y, isPhase=False):
        """o--->x
           |
           y
           x1<x2, y1<y2, (x,y) in between
        """
        r1 = self.linear_interpolation_1d(x1, x2, fx11, fx21, x, isPhase)
        r2 = self.linear_interpolation_1d(x1, x2, fx12, fx22, x, isPhase)
        p = self.linear_interpolation_1d(y1, y2, r1, r2, y, isPhase)
        return p

    def bilinear_interpolation_3d(self, x1, x2, y1, y2, z1, z2, fx111, fx211, fx121, fx221, fx112, fx212, fx122, fx222, x, y, z,
                                  isPhase=False):
        "Pay attention to the order of input parameters. This bug causes me around 12 hours to debug."
        c1 = self.bilinear_interpolation_2d(x1, x2, y1, y2, fx111, fx211, fx121, fx221, x, y, isPhase)
        c2 = self.bilinear_interpolation_2d(x1, x2, y1, y2, fx112, fx212, fx122, fx222, x, y, isPhase)
        p = self.linear_interpolation_1d(z1, z2, c1, c2, z, isPhase)
        return p

    def grid_sampler_2d(self, img, grid, isPhase=False):
        "ignore the image boudnary"
        assert img.shape == grid.shape[:-1]
        # get grid shape
        H, W, _ = grid.shape

        # get pixel coordinates
        x = grid[..., 0]
        y = grid[..., 1]

        # get pixel indices
        x0 = torch.floor(x).long()
        x1 = x0 + 1
        y0 = torch.floor(y).long()
        y1 = y0 + 1

        x0_ = torch.clamp(x0, min=0, max=W - 2)
        x1_ = x0_ + 1
        y0_ = torch.clamp(y0, min=0, max=H - 2)
        y1_ = y0_ + 1

        # get pixel values
        I00 = img[y0_, x0_]
        I10 = img[y1_, x0_]
        I01 = img[y0_, x1_]
        I11 = img[y1_, x1_]

        outsider = (x < 0) | (x > W - 1) | (y < 0) | (y > H - 1)
        I00[outsider] = 0
        I10[outsider] = 0
        I01[outsider] = 0
        I11[outsider] = 0

        out = self.bilinear_interpolation_2d(x0, x1, y0, y1, I00, I10, I01, I11, x, y, isPhase=isPhase)
        return out

    def grid_sampler_3d(self, img, grid, isPhase=False):
        """grid[...,0] is x position, grid[...,1] is y position, grid[...,2] is z position. xyz are in lab coordinate system.
        x corresponds to left-right (LR), y corresponds to front-back (AP), z corresponds to up-down (IS).
        Illustrator: https://drive.google.com/file/d/1XfmADDOx5kVl-u-P975fJ-r7txGUbvdn/view?usp=share_link
        Interpolation ignores the image boundary for simplicity."""
        assert img.shape == grid.shape[:-1]
        # get grid shape
        H,W,D, _ = grid.shape

        # get pixel coordinates
        x = grid[..., 0]
        y = grid[..., 1]
        z = grid[..., 2]

        # get pixel indices
        x0 = torch.floor(x).long()
        x1 = x0 + 1
        y0 = torch.floor(y).long()
        y1 = y0 + 1
        z0 = torch.floor(z).long()
        z1 = z0 + 1

        x0_ = torch.clamp(x0, 0, W-2)
        x1_ = x0_ + 1
        y0_ = torch.clamp(y0, 0, H-2)
        y1_ = y0_ + 1
        z0_ = torch.clamp(z0, 0, D-2)
        z1_ = z0_ + 1

        # get pixel values
        I000 = img[y0_, x0_, z0_]
        I100 = img[y1_, x0_, z0_]
        I010 = img[y0_, x1_, z0_]
        I110 = img[y1_, x1_, z0_]
        I001 = img[y0_, x0_, z1_]
        I101 = img[y1_, x0_, z1_]
        I011 = img[y0_, x1_, z1_]
        I111 = img[y1_, x1_, z1_]

        outsiders = (x < 0) | (x > W - 1) | (y < 0) | (y > H - 1) | (z < 0) | (z > D - 1)
        I000[outsiders] = 0
        I100[outsiders] = 0
        I010[outsiders] = 0
        I110[outsiders] = 0
        I001[outsiders] = 0
        I101[outsiders] = 0
        I011[outsiders] = 0
        I111[outsiders] = 0

        out = self.bilinear_interpolation_3d(x0, x1, y0, y1, z0, z1, I000, I100, I010, I110, I001, I101, I011, I111, x, y, z, isPhase=isPhase)
        return out

    def grid_sampler_3d_in_batch(self, img, grid, isPhase=False):
        ndim = len(img.shape)
        assert ndim == 5
        B, C, H, W, D = img.shape
        assert img.shape[-3:] == grid.shape[1:-1] # grid: B, H, W, D, 3
        assert img.shape[0] == grid.shape[0]
        warped = torch.empty_like(img)
        for b in range(B):
            for c in range(C):
                warped[b, c] = self.grid_sampler_3d(img[b, c], grid[b], isPhase=isPhase)
        return warped



    def grid_sampler_2d_in_batch(self, img, grid, isPhase=False):
        shape = img.shape
        ndim = len(shape)
        assert ndim == 4
        B, C, H, W = shape
        assert shape[-2:] == grid.shape[1:-1]
        warped = torch.empty_like(img)
        for b in range(B):
            for c in range(C):
                warped[b, c] = self.grid_sampler_2d(img[b, c], grid[b], isPhase=isPhase)
        return warped

def flow_to_grid_3d(disp):
    assert len(disp.shape) == 5 # B, H, W, D, 3
    b,h,w,d,_ = disp.shape
    grid_h, grid_w, grid_d = torch.meshgrid(
        [torch.linspace(0, h - 1, h), torch.linspace(0, w - 1, w), torch.linspace(0, d - 1, d)])  # compute grid
    grid = torch.stack([grid_w, grid_h, grid_d], dim=3)
    grid = repeat(grid, 'h w d c -> b h w d c', b=b)
    sample_grid = disp + grid.type_as(disp)
    return sample_grid

def flow_to_grid_2d(disp):
    assert len(disp.shape) == 4 # B, H, W, 3
    b,h,w,_ = disp.shape
    grid_h, grid_w = torch.meshgrid(
        [torch.linspace(0, h - 1, h), torch.linspace(0, w - 1, w)])  # compute grid
    grid = torch.stack([grid_w, grid_h], dim=2)
    grid = repeat(grid, 'h w c -> b h w c', b=b)
    sample_grid = disp + grid
    return sample_grid

def mytest_if_wrap_function_is_differentiable():
    # prove mod function is differentiable by torch.autograd
    x = torch.range(0, 20, 0.1, requires_grad=True)
    y = torch.remainder(x, 2 * torch.pi)
    # y = torch.fmod(x, 2*torch.pi)

    y.retain_grad()
    z = torch.sum(2 * y)
    z.retain_grad()
    z.backward()
    print(z)
    print(z.grad)  # 1
    print(y.grad)  # 2
    print(x.grad)  # 1*2

######################### 2D Warp ##########################
def mesh_grid(B, H, W):
    # mesh grid
    x_base = torch.arange(0, W).repeat(B, H, 1)  # BHW
    y_base = torch.arange(0, H).repeat(B, W, 1).transpose(1, 2)  # BHW

    base_grid = torch.stack([x_base, y_base], 1)  # B2HW
    return base_grid

def norm_grid(v_grid):
    _, _, H, W = v_grid.size()

    # scale grid to [-1,1]
    v_grid_norm = torch.zeros_like(v_grid)
    v_grid_norm[:, 0, :, :] = 2.0 * v_grid[:, 0, :, :] / (W - 1) - 1.0
    v_grid_norm[:, 1, :, :] = 2.0 * v_grid[:, 1, :, :] / (H - 1) - 1.0
    return v_grid_norm  # BHW2

def flow_warp(x, flow12, pad='zeros', mode='bilinear'):
    B, _, H, W = x.size()
    if flow12.shape[-1] == 2:
        flow12 = rearrange(flow12, 'b h w c -> b c h w')

    base_grid = mesh_grid(B, H, W).type_as(x)  # B2HW

    v_grid = norm_grid(base_grid) + flow12  # BHW2
    v_grid = rearrange(v_grid, 'b c h w -> b h w c')
    im1_recons = nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad, align_corners=True)
    return im1_recons


######################### 3D Warp ##########################
def norm_grid_3d(v_grid):
    _, _, H, W, D = v_grid.size()

    # scale grid to [-1,1]
    v_grid_norm = torch.zeros_like(v_grid)
    v_grid_norm[:, 0, :, :, :] = 2.0 * v_grid[:, 0, :, :, :] / (W - 1) - 1.0 # the same as voxelmorph. Note divided by W-1 instead of W. LKunet does it wrong!
    v_grid_norm[:, 1, :, :, :] = 2.0 * v_grid[:, 1, :, :, :] / (H - 1) - 1.0
    v_grid_norm[:, 2, :, :, :] = 2.0 * v_grid[:, 2, :, :, :] / (D - 1) - 1.0

    return v_grid_norm  # B3HWD

def mesh_grid_3d(B, H, W, D):
    # mesh grid
    grid_h, grid_w, grid_d = torch.meshgrid(
        [torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W), torch.linspace(0, D - 1, D)])  # compute grid
    base_grid = torch.stack([grid_w, grid_h, grid_d], dim=0).repeat(B, 1, 1, 1, 1)  # Bx3xHxWxD

    return base_grid

def flow_warp_3d(x, flow12, pad='zeros', mode='bilinear'):
    B, _, H, W, D = x.size()
    if flow12.shape[-1] == 3:
        flow12 = rearrange(flow12, 'b h w d c -> b c h w d')
    base_grid = mesh_grid_3d(B, H, W, D).type_as(x)  # B3HWD

    v_grid = norm_grid_3d(base_grid) + flow12  # BHWD3 (x,y,z). flow ranges from [-1,1], which is bounded by network's output layer.
    v_grid = rearrange(v_grid, 'b c h w d -> b h w d c')
    v_grid = v_grid[...,[2,0,1]] # (z,x,y)
    im1_recons = nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad, align_corners=True)
    return im1_recons

def mytest_if_grid_sampler_3d_in_batch_the_same_as_flow_warp_3d():
    torch.manual_seed(1)
    B, C, H, W, D = 1, 1, 8, 9, 10
    img = torch.rand(B, C, H, W, D)
    flow = torch.rand(B, H,W,D, 3)
    # flow = torch.zeros(B, H, W, D, 3)
    # flow[:, :, :, 0] = -0.3
    # flow[:, :, :, 1] = -0.4
    # flow[:, :, :, 2] = -0.5

    grid = flow_to_grid_3d(flow)
    warped = grid_sampler_3d_in_batch(img, grid)
    warped2 = flow_warp_3d(img, flow)
    print("error=", torch.abs(warped-warped2).mean())
    assert torch.allclose(warped, warped2, atol=1e-5)

def mytest_if_grid_sampler_2d_in_batch_the_same_as_flow_warp_2d():
    B, C, H, W = 2, 1, 8,9
    img = torch.rand(B, C, H, W)
    flow = torch.rand(B, H, W, 2)
    grid = flow_to_grid_2d(flow)
    warped = grid_sampler_2d_in_batch(img, grid)
    warped2 = flow_warp(img, flow)
    assert torch.allclose(warped, warped2, atol=1e-5)


def plot3view(img, slice=3):
    fig, axs = plt.subplots(1, 3, figsize=(15, 15))
    axs[0].imshow(img[:, :, slice], cmap='gray', origin='lower')
    axs[0].set_xlabel("x");
    axs[0].set_ylabel("y");
    axs[0].set_title("axial")

    axs[1].imshow(img[:, slice, :].T, cmap='gray', origin='lower')
    axs[1].set_xlabel("y");
    axs[1].set_ylabel("z");
    axs[1].set_title("sag")

    axs[2].imshow(img[slice, :, :].T, cmap='gray', origin='lower')
    axs[2].set_xlabel("x");
    axs[2].set_ylabel("z");
    axs[2].set_title("coronal")
    plt.show()



class VxmSpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed # TODO
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)





