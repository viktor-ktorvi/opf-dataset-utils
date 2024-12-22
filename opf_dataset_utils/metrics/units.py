from enum import StrEnum

from torch import Tensor


class UnitTypes(StrEnum):
    PER_UNIT = "per-unit"
    BASE_SI = "base_SI_unit"
    KILO = "kilo"
    MEGA = "mega"
    GIGA = "giga"


def convert_unit(values_per_unit: Tensor, base_mega: Tensor, unit: str) -> Tensor:
    """
    Convert a per-unit value into its value in the specified unit.
    Parameters
    ----------
    values_per_unit: Tensor
        Value in per-unit.
    base_mega: Tensor
        The base values of the corresponding per-unit system given in 'mega' units.
    unit: str
        The desired unit.

    Returns
    -------
    values_unit: Tensor
        Values converted to the specified unit.

    Raises
    ------
    ValueError:
        If the unit is not supported.
    """
    if unit == UnitTypes.PER_UNIT:
        return values_per_unit

    values_mega = values_per_unit * base_mega

    if unit == UnitTypes.MEGA:
        return values_mega

    if unit == UnitTypes.GIGA:
        return values_mega / 1e3

    if unit == UnitTypes.KILO:
        return values_mega * 1e3

    if unit == UnitTypes.BASE_SI:
        return values_mega * 1e6

    raise ValueError(f"The unit {unit} is not supported. Expected one of {[str(u) for u in UnitTypes]}")
