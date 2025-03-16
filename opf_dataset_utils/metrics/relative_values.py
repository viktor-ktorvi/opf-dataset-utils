from enum import StrEnum

from torch import Tensor


class ValueTypes(StrEnum):
    ABSOLUTE = "absolute"
    RELATIVE = "relative"


def calculate_relative_values(numerator: Tensor, denominator: Tensor, epsilon: float = 0.001) -> Tensor:
    """
    Calculate relative values in percentage points %.

    Parameters
    ----------
    numerator: Tensor
        Numerator.
    denominator: Tensor
        Denominator.
    epsilon: float
        A values added to the denominator to prevent divisions by zero or small values.

    Returns
    -------
    relative_values: Tensor
        Relative values in [%].
    """
    numerator = numerator.abs()
    denominator = denominator.abs()

    relative_values = numerator / (denominator + epsilon) * 100.0

    return relative_values
