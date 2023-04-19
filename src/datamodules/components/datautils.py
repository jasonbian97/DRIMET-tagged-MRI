import numpy as np
import torch
from src.models.components.Jacobian import jacobian_determinant_tensor, jacobian_determinant_numpy
from einops import rearrange

def compute_det_auc(flow, weight=None):
    if flow.shape[0]==3:
        flow = rearrange(flow, 'c h w d -> h w d c')
    det = jacobian_determinant_numpy(flow)
    deviation = np.abs(det.flatten() - 1)
    count, bins_count = np.histogram(deviation, bins=100, range=(0, 1.0), weights=weight.flatten())
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    auc = np.sum(cdf) * (1 / 100.)
    return auc, cdf

def compute_det_auc_ten(flow, weight=None): # flow: BCHWD
    flow = rearrange(flow, 'b c h w d -> b h w d c')
    det = jacobian_determinant_tensor(flow)
    deviation = torch.abs(det.flatten() - 1)
    count, bins_count = torch.histogram(deviation.cpu(), bins=100, range=(0, 1.0), weight=weight.flatten().cpu())
    pdf = count / count.sum()
    cdf = torch.cumsum(pdf, dim=0)
    auc = torch.sum(cdf) * (1 / 100.)
    return auc.item()

def compute_neg_det(flow, weight=None):
    if flow.shape[0]==3:
        flow = rearrange(flow, 'c h w d -> h w d c')
    det = jacobian_determinant_numpy(flow)
    neg_det = np.sum(det<0) if weight is None else np.sum((det<0)*weight)
    return neg_det

def flow_mag_ten(flow):
    return torch.sqrt(torch.sum(flow**2, dim=1, keepdim=True))

def metr_dict_2_cpu(metr_dict):
    "recursively convert all tensors in metr_dict to cpu"
    for k, v in metr_dict.items():
        if isinstance(v, dict):
            metr_dict[k] = metr_dict_2_cpu(v)
        elif isinstance(v, torch.Tensor):
            metr_dict[k] = v.cpu().item()
    return metr_dict

def list_tensor_2_numpy(list_tensor):
    "convert a list of tensors to a list of numpy arrays"
    list_numpy = []
    for tensor in list_tensor:
        list_numpy.append(tensor.cpu().numpy())
    return list_numpy

def compute_neg_det_frac_ten(flow, weight=None):
    if len(flow.shape)==4:
        flow = rearrange(flow, 'b c h w -> b h w c')
    elif len(flow.shape)==5:
        flow = rearrange(flow, 'b c h w d -> b h w d c')
    else:
        raise ValueError("flow shape must be BCHW or BCHWD")
    det = jacobian_determinant_tensor(flow)
    neg_det = torch.sum(det<0) if weight is None else torch.sum((det<0)*weight)
    size = np.prod(det.shape)
    return (neg_det/size).item()

def cvt_sin_cos_to_phase(sin_img, cos_img):
    phase_img = np.arctan2(sin_img, cos_img)
    return phase_img

def cvt_phase_to_sin_cos(phase_img):
    sin_img = np.sin(phase_img)
    cos_img = np.cos(phase_img)
    return sin_img, cos_img

# torch version of cvt_sin_cos_to_phase, cvt_phase_to_sin_cos
def cvt_sin_cos_to_phase_torch(sin_img, cos_img):
    phase_img = torch.atan2(sin_img, cos_img)
    return phase_img

def cvt_phase_to_sin_cos_torch(phase_img):
    sin_img = torch.sin(phase_img)
    cos_img = torch.cos(phase_img)
    return sin_img, cos_img
