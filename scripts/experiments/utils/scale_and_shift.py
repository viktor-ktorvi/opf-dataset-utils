import torch
from torch import Tensor


def scale_and_shift(x: Tensor, x_min: Tensor, x_max: Tensor) -> Tensor:
    """
    Scale and shift using a sigmoid.

    Parameters
    ----------
    x: Tensor
    x_min: Tensor
    x_max: Tensor

    Returns
    -------
    scaled_and_shifted_value: Tensor
    """
    return torch.sigmoid(x) * (x_max - x_min) + x_min
