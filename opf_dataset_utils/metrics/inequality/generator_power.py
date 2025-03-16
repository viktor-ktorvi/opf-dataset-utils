from typing import Optional

from torch import Tensor
from torch_geometric.data import HeteroData

from opf_dataset_utils.enumerations import GridGeneratorIndices, NodeTypes
from opf_dataset_utils.errors.inequality.generator_power import (
    calculate_lower_active_power_errors,
    calculate_lower_reactive_power_errors,
    calculate_upper_active_power_errors,
    calculate_upper_reactive_power_errors,
)
from opf_dataset_utils.metrics.aggregation import AggregatorMetric
from opf_dataset_utils.metrics.inequality.bound_types import BoundTypes
from opf_dataset_utils.metrics.power import PowerTypes
from opf_dataset_utils.metrics.relative_values import (
    ValueTypes,
    calculate_relative_values,
)
from opf_dataset_utils.metrics.units import convert_unit


class GeneratorPowerInequalityError(AggregatorMetric):
    """A metric for voltage angle difference inequality violations."""

    is_differentiable: Optional[bool] = True
    higher_is_better: Optional[bool] = False
    full_state_update: bool = True

    def __init__(self, aggr: str, bound_type: str, value_type: str, power_type: str, unit: str = "per-unit", **kwargs):
        super().__init__(aggr, **kwargs)
        self.value_type = value_type
        self.bound_type = bound_type
        self.unit = unit
        self.power_type = power_type

        if value_type not in list(ValueTypes):
            raise ValueError(
                f"Value type '{value_type}' is not supported. Expected one of {[vt for vt in ValueTypes]}."
            )

        if bound_type not in list(BoundTypes):
            raise ValueError(
                f"Bound type '{bound_type}' is not supported. Expected one of {[bt for bt in BoundTypes]}."
            )

        if power_type not in [PowerTypes.ACTIVE, PowerTypes.REACTIVE]:
            raise ValueError(
                f"Power type '{power_type}' is not supported. Expected one of {[PowerTypes.ACTIVE, PowerTypes.REACTIVE]}."
            )

    def update(self, batch: HeteroData, predictions: dict[str, Tensor]):
        errors_pu = None
        if self.bound_type == BoundTypes.UPPER:
            if self.power_type == PowerTypes.ACTIVE:
                errors_pu = calculate_upper_active_power_errors(batch, predictions)

            if self.power_type == PowerTypes.REACTIVE:
                errors_pu = calculate_upper_reactive_power_errors(batch, predictions)

        if self.bound_type == BoundTypes.LOWER:
            if self.power_type == PowerTypes.ACTIVE:
                errors_pu = calculate_lower_active_power_errors(batch, predictions)

            if self.power_type == PowerTypes.REACTIVE:
                errors_pu = calculate_lower_reactive_power_errors(batch, predictions)

        errors_pu = errors_pu.abs()

        if self.value_type == ValueTypes.ABSOLUTE:
            baseMVA_per_generator = batch.x_dict[NodeTypes.GENERATOR][:, GridGeneratorIndices.TOTAL_BASE_MVA]
            return super().update(convert_unit(errors_pu, baseMVA_per_generator, self.unit))

        if self.value_type == ValueTypes.RELATIVE:
            power_max = None
            power_min = None

            if self.power_type == PowerTypes.ACTIVE:
                power_max = batch.x_dict[NodeTypes.GENERATOR][:, GridGeneratorIndices.ACTIVE_POWER_MAX]
                power_min = batch.x_dict[NodeTypes.GENERATOR][:, GridGeneratorIndices.ACTIVE_POWER_MIN]

            if self.power_type == PowerTypes.REACTIVE:
                power_max = batch.x_dict[NodeTypes.GENERATOR][:, GridGeneratorIndices.REACTIVE_POWER_MAX]
                power_min = batch.x_dict[NodeTypes.GENERATOR][:, GridGeneratorIndices.REACTIVE_POWER_MIN]

            return super().update(calculate_relative_values(errors_pu, power_max - power_min))
