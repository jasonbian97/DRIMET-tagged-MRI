import torch
import torch.nn.functional as F
import numpy as np
import math
import torch.nn as nn
from einops import rearrange
from monai.losses import GlobalMutualInformationLoss
from src.models.components.SpatialWarp import scale_flow


class NCC:
    """
    Local (over window) normalized cross correlation loss.
    bzx modified: enable multi-channel input: trick put channel dimention to the first dimention (N). Take each channle as a sperate image
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_pred, y_true):
        device = y_pred.device
        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, nb_feats, *vol_shape]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [self.win, ] * ndims

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to(device)

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)

    def loss_multichannel(self, y_pred, y_true):
        """
        Compute the NCC loss for each channel and return the mean. A more general form.
        """
        loss = 0
        for i in range(y_pred.shape[1]):
            loss += self.loss(y_pred[:, i:i + 1], y_true[:, i:i + 1])
        return loss / y_pred.shape[1]


class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_pred, y_true):
        return torch.mean((y_true - y_pred) ** 2)


class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice


class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, y_pred):
        if len(y_pred.shape) == 5:  # 3D
            dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
            dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
            dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

            if self.penalty == 'l2':
                dy = dy * dy
                dx = dx * dx
                dz = dz * dz

            d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
            grad = d / 3.0

        elif len(y_pred.shape) == 4:  # 2D
            dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
            dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])
            if self.penalty == 'l2':
                dy = dy * dy
                dx = dx * dx

            d = torch.mean(dx) + torch.mean(dy)
            grad = d / 2.0
        else:
            raise ValueError('Expected 4D or 5D tensor, got %dD tensor instead.' % len(y_pred.shape))

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad


from src.models.components.Jacobian import jacobian_determinant_tensor
from typing import Callable, Optional, Sequence, Union


class Incompress(nn.Module):
    """Incompressibliity loss for vector fields
    """

    def __init__(self, form="abslog", abs="l1"):
        """form can be {"abslog", "abslog_diffeo", "minus1", "reciprocal"}.
        abs can be {"l1", "l2"}."""
        super().__init__()
        assert form in ["abslog", "abslog_diffeo", "minus1", "reciprocal"]
        assert abs in ["l1", "l2"]

        self.form = form
        self.jacobianeer = jacobian_determinant_tensor
        self.abs = torch.abs if abs == "l1" else torch.square

    def forward(self, vec, weight=1.0):
        eps = 1e-7
        if self.form == "abslog":
            Jdet = self.jacobianeer(vec)
            return (self.abs(torch.log(F.relu(Jdet) + eps)) * weight).mean()
        elif self.form == "abslog_diffeo":
            Jdet = self.jacobianeer(vec)
            return (self.abs(torch.log(F.relu(Jdet) + eps)) * weight).mean() + F.relu(
                -Jdet).mean()  # this doesnt penalize J<0
        elif self.form == "minus1":
            one = torch.tensor(1.0).type_as(vec)
            Jdet = self.jacobianeer(vec)
            return (self.abs(Jdet - one) * weight).mean()
        elif self.form == "reciprocal":
            J = self.jacobianeer(vec)
            return (J + 1 / J - 2).mean()
        else:
            raise ValueError("incompress loss form not recognized")


class CatLoss(nn.Module):
    def __init__(self,
                 image_loss: str = "mse",
                 w_sm: float = 0.1,
                 incompress_form: Optional[str] = "abslog",
                 incompress_abs: Optional[str] = "l1",
                 w_in: Optional[float] = 0.1,
                 ncc_win: int = 9,
                 ):
        """incompress_form can be {"abslog", "minus1", "reciprocal"}.
        incompress_abs can be {"l1", "l2"}."""

        super().__init__()

        self.w_sm = w_sm
        self.w_in = w_in

        if image_loss == 'ncc':
            self.image_loss_func = NCC(win=ncc_win).loss_multichannel
        elif image_loss == 'mse':
            self.image_loss_func = MSE().loss
        elif image_loss == 'mi':
            self.image_loss_func = GlobalMutualInformationLoss()
        elif image_loss == 'l1':
            self.image_loss_func = nn.L1Loss(reduction='mean')
        else:
            raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % image_loss)

        self.sm_loss_func = Grad('l2', loss_mult=None).loss

        if incompress_form and incompress_abs and w_in:
            self.incompress_loss_func = Incompress(form=incompress_form, abs=incompress_abs)
        else:
            self.incompress_loss_func = None

    def forward(self, fixed, moved, flow, weight=1.0):
        """weight: should be B,1,H,W,D, e.g., mag image"""
        similarity_loss = self.image_loss_func(fixed, moved) * 1.0
        smoothness_loss = self.sm_loss_func(flow) * self.w_sm
        H, W, D = flow.shape[2:]
        scaled_flow = scale_flow(flow, (H / 2., W / 2., D / 2.))
        incompress_loss = self.incompress_loss_func(scaled_flow,
                                                    weight) * self.w_in if self.incompress_loss_func else torch.tensor(
            0.).type_as(flow)
        tot_loss = similarity_loss + smoothness_loss + incompress_loss

        return tot_loss, similarity_loss.item(), smoothness_loss.item(), incompress_loss.item()


class MSE_SM(nn.Module):
    def __init__(self,
                 w_sm: float = 0.1,
                 ):
        super().__init__()
        self.w_sm = w_sm
        self.image_loss_func = MSE().loss
        self.sm_loss_func = Grad('l2', loss_mult=None).loss

    def forward(self, fixed, moved, flow, weight=1.0):
        similarity_loss = self.image_loss_func(fixed, moved) * 1.0
        smoothness_loss = self.sm_loss_func(flow) * self.w_sm
        tot_loss = similarity_loss + smoothness_loss
        return tot_loss, similarity_loss.item(), smoothness_loss.item()
