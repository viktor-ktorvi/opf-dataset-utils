from typing import Optional

from torch import Tensor
from torch_geometric.data import HeteroData

from opf_dataset_utils.enumerations import (
    GridGeneratorIndices,
    NodeTypes,
    SolutionGeneratorIndices,
)
from opf_dataset_utils.metrics.aggregation import AggregatorMetric
from opf_dataset_utils.metrics.power import PowerTypes
from opf_dataset_utils.metrics.relative_values import (
    ValueTypes,
    calculate_relative_values,
)
from opf_dataset_utils.metrics.units import convert_unit


class GeneratorPowerError(AggregatorMetric):
    """
    A metric that measures absolute and relative errors of the generator power predictions.
    """

    higher_is_better: Optional[bool] = False
    is_differentiable: Optional[bool] = True
    full_state_update: bool = True

    def __init__(self, aggr: str, value_type: str, power_type: str, unit: str = "per-unit", **kwargs):
        super().__init__(aggr, **kwargs)
        self.power_type = power_type
        self.unit = unit
        self.value_type = value_type

        if power_type not in [PowerTypes.ACTIVE, PowerTypes.REACTIVE]:
            raise ValueError(
                f"Power type '{power_type}' is not supported. Expected one of {[PowerTypes.ACTIVE, PowerTypes.REACTIVE]}."
            )

        if value_type not in list(ValueTypes):
            raise ValueError(
                f"Value type '{value_type}' is not supported. Expected one of {[vt for vt in ValueTypes]}."
            )

    def update(self, batch: HeteroData, predictions: dict[str, Tensor]):
        powers_pred_pu = None
        powers_target_pu = None
        if self.power_type == PowerTypes.ACTIVE:
            powers_pred_pu = predictions[NodeTypes.GENERATOR][:, SolutionGeneratorIndices.ACTIVE_POWER]
            powers_target_pu = batch.y_dict[NodeTypes.GENERATOR][:, SolutionGeneratorIndices.ACTIVE_POWER]

        if self.power_type == PowerTypes.REACTIVE:
            powers_pred_pu = predictions[NodeTypes.GENERATOR][:, SolutionGeneratorIndices.REACTIVE_POWER]
            powers_target_pu = batch.y_dict[NodeTypes.GENERATOR][:, SolutionGeneratorIndices.REACTIVE_POWER]

        abs_errors_pu = (powers_pred_pu - powers_target_pu).abs()

        if self.value_type == ValueTypes.ABSOLUTE:
            baseMVA_per_generator = batch.x_dict[NodeTypes.GENERATOR][:, GridGeneratorIndices.TOTAL_BASE_MVA]
            return super().update(convert_unit(abs_errors_pu, baseMVA_per_generator, self.unit))

        if self.value_type == ValueTypes.RELATIVE:
            return super().update(calculate_relative_values(abs_errors_pu, powers_target_pu))


class GeneratorPower(AggregatorMetric):
    """
    A metric that measures the generator-level powers in the grid.
    """

    is_differentiable: Optional[bool] = True
    full_state_update: bool = True

    def __init__(self, aggr: str, power_type: str, unit: str = "per-unit", **kwargs):
        super().__init__(aggr, **kwargs)
        self.power_type = power_type
        self.unit = unit

        if power_type not in [PowerTypes.ACTIVE, PowerTypes.REACTIVE]:
            raise ValueError(
                f"Power type '{power_type}' is not supported. Expected one of {[PowerTypes.ACTIVE, PowerTypes.REACTIVE]}."
            )

    def update(self, batch: HeteroData, predictions: dict[str, Tensor]):
        powers_pred_pu = None
        if self.power_type == PowerTypes.ACTIVE:
            powers_pred_pu = predictions[NodeTypes.GENERATOR][:, SolutionGeneratorIndices.ACTIVE_POWER]

        if self.power_type == PowerTypes.REACTIVE:
            powers_pred_pu = predictions[NodeTypes.GENERATOR][:, SolutionGeneratorIndices.REACTIVE_POWER]

        baseMVA_per_generator = batch.x_dict[NodeTypes.GENERATOR][:, GridGeneratorIndices.TOTAL_BASE_MVA]
        return super().update(convert_unit(powers_pred_pu, baseMVA_per_generator, self.unit))
