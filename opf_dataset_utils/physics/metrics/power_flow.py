from typing import Optional

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from opf_dataset_utils.enumerations import NodeTypes
from opf_dataset_utils.physics.errors.power_flow import calculate_power_flow_errors
from opf_dataset_utils.physics.metrics.aggregation import AggregatorMetric
from opf_dataset_utils.physics.metrics.power import calculate_power_type
from opf_dataset_utils.physics.metrics.relative_values import (
    ValueTypes,
    calculate_relative_values,
)
from opf_dataset_utils.physics.metrics.units import convert_unit
from opf_dataset_utils.physics.power import calculate_bus_powers


class PowerFlowError(AggregatorMetric):
    """
    An absolute power flow error metric with specifiable aggregation, power type, unit, and value type.
    """

    is_differentiable: Optional[bool] = True
    higher_is_better: Optional[bool] = False
    full_state_update: bool = True

    complex_powers: Tensor
    complex_power_flow_errors: Tensor

    def __init__(self, aggr: str, power_type: str, value_type: str, unit: str = "per-unit", **kwargs):
        super().__init__(aggr, **kwargs)
        self.power_type = power_type
        self.unit = unit
        self.value_type = value_type

        if value_type not in list(ValueTypes):
            raise ValueError(
                f"Value type '{value_type}' is not supported. Expected one of {[vt for vt in ValueTypes]}."
            )

        self.add_state("complex_powers", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("complex_power_flow_errors", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, batch: HeteroData, predictions: dict[str, Tensor]):
        self.complex_power_flow_errors = calculate_power_flow_errors(batch, predictions)

        errors_pu = calculate_power_type(self.complex_power_flow_errors, self.power_type)
        errors_pu = errors_pu.abs()

        if self.value_type == ValueTypes.ABSOLUTE:
            baseMVA_per_bus = batch.x[batch.batch_dict[NodeTypes.BUS]]

            return super().update(convert_unit(errors_pu, baseMVA_per_bus, self.unit))

        if self.value_type == ValueTypes.RELATIVE:
            self.complex_powers = calculate_bus_powers(batch, predictions)
            powers_pu = calculate_power_type(self.complex_powers, self.power_type)
            powers_pu = powers_pu.abs()

            return super().update(calculate_relative_values(errors_pu, powers_pu))
