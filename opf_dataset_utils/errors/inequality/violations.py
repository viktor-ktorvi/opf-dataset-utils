import torch
from torch import Tensor


def calculate_upper_violations(values: Tensor, values_max: Tensor) -> Tensor:
    """
    Return the difference between the value and its maximum value if the value is larger than the maximum value,
    otherwise return 0.

    Parameters
    ----------
    values: Tensor
        Values.
    values_max: Tensor
        Corresponding maximum values.
    Returns
    -------
    upper_violations: Tensor
        Upper violations.
    """
    return torch.maximum(torch.zeros_like(values), values - values_max)


def calculate_lower_violations(values: Tensor, values_min: Tensor) -> Tensor:
    """
    Return the difference between the value and its minimum value if the value is smaller than the minimum value,
    otherwise return 0.

    Parameters
    ----------
    values: Tensor
        Values.
    values_min: Tensor
        Corresponding minimum values.

    Returns
    -------
    lower_violations: Tensor
        Lower violations.
    """
    return torch.minimum(torch.zeros_like(values), values - values_min)
