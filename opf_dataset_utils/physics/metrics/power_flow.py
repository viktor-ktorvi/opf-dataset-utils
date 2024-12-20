from enum import StrEnum
from typing import Optional

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from opf_dataset_utils.enumerations import NodeTypes
from opf_dataset_utils.physics.errors.power_flow import calculate_power_flow_errors
from opf_dataset_utils.physics.metrics.aggregation import AggregatorMetric
from opf_dataset_utils.physics.metrics.units import convert_unit


class PowerTypes(StrEnum):
    APPARENT = "apparent"
    ACTIVE = "active"
    REACTIVE = "reactive"


def calculate_power_type(complex_powers: Tensor, power_type: str) -> Tensor:
    """
    Calculate the specified power type from complex power values.
    Parameters
    ----------
    complex_powers: Tensor
        Complex power values.
    power_type: str
        Power type.

    Returns
    -------
    power: Tensor
        Power.
    Raises
    ------
    ValueError:
        If the unit is not supported.
    """
    if power_type == PowerTypes.ACTIVE:
        return complex_powers.real

    if power_type == PowerTypes.REACTIVE:
        return complex_powers.imag

    if power_type == PowerTypes.APPARENT:
        return complex_powers.abs()

    raise ValueError(f"Power type '{power_type}' is not supported. Expected one of {[str(pt) for pt in PowerTypes]}")


class AbsolutePowerFlowError(AggregatorMetric):
    """
    An absolute power flow error metric with specifiable aggregation, power type, and unit.
    """
    is_differentiable: Optional[bool] = True
    higher_is_better: Optional[bool] = False
    full_state_update: bool = True

    complex_power_flow_errors: Tensor

    def __init__(
            self,
            aggr: str,
            power_type: str,
            unit: str,
            **kwargs
    ):
        super().__init__(aggr, **kwargs)
        self.power_type = power_type
        self.unit = unit
        self.add_state("complex_power_flow_errors", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, batch: HeteroData, predictions: dict[str, Tensor]):
        self.complex_power_flow_errors = calculate_power_flow_errors(batch, predictions)

        errors = calculate_power_type(self.complex_power_flow_errors, self.power_type)
        errors = errors.abs()

        baseMVA_per_bus = batch.x[batch.batch_dict[NodeTypes.BUS]]

        super().update(
            convert_unit(errors, baseMVA_per_bus, self.unit)
        )
