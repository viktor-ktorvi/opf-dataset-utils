from enum import StrEnum

from torch import Tensor


class ValueTypes(StrEnum):
    ABSOLUTE = "absolute"
    RELATIVE = "relative"


def calculate_relative_values(numerator: Tensor, denominator: Tensor) -> Tensor:
    """
    Calculate relative values in percentage points %. Exclude values where the denominator is 0.
    Parameters
    ----------
    numerator: Tensor
        Numerator.
    denominator: Tensor
        Denominator.

    Returns
    -------
    relative_values: Tensor
        Relative values in [%].
    """
    numerator = numerator.abs()
    denominator = denominator.abs()

    mask = denominator > 0.0

    relative_values = numerator[mask] / denominator[mask] * 100.0

    return relative_values
