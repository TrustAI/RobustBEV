import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


def factor2tensor(factor, device, dtype, img_shape):
    """
    Adopted from KORNIA
    https://kornia.readthedocs.io/en/latest/_modules/kornia/enhance/adjust.html
    """
    if isinstance(factor, float):
        factor = torch.as_tensor(factor, device=device, dtype=dtype)
    elif isinstance(factor, Tensor):
        factor = factor.to(device, dtype)
    while len(factor.shape) != len(img_shape):
        factor = factor[..., None]
    return factor


class AffineTransf(nn.Module):

    def __init__(self, theta):
        super(AffineTransf, self).__init__()
        self.theta = theta.view(1,2,3)
        
    def forward(self, img):
        batch_size, channel, height, width = img.size()

        grid = F.affine_grid(self.theta, torch.Size((
            batch_size, channel, height, width)))
        transfored_img = F.grid_sample(img, grid)
        return transfored_img