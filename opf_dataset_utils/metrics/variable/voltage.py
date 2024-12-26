from typing import Optional

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from opf_dataset_utils.enumerations import GridBusIndices, NodeTypes, SolutionBusIndices
from opf_dataset_utils.metrics.aggregation import AggregatorMetric
from opf_dataset_utils.metrics.relative_values import (
    ValueTypes,
    calculate_relative_values,
)
from opf_dataset_utils.metrics.units import AngleUnits, convert_unit


class VoltageMagnitudeError(AggregatorMetric):
    """
    A metric that measures absolute and relative errors of the bus voltage magnitude predictions.
    """

    is_differentiable: Optional[bool] = True
    higher_is_better: Optional[bool] = False
    full_state_update: bool = True

    def __init__(self, aggr: str, value_type: str, unit: str = "per-unit", **kwargs):
        super().__init__(aggr, **kwargs)
        self.unit = unit
        self.value_type = value_type

        if value_type not in list(ValueTypes):
            raise ValueError(
                f"Value type '{value_type}' is not supported. Expected one of {[vt for vt in ValueTypes]}."
            )

    def update(self, batch: HeteroData, predictions: dict[str, Tensor]):
        magnitudes_pred_pu = predictions[NodeTypes.BUS][:, SolutionBusIndices.VOLTAGE_MAGNITUDE]
        magnitudes_target_pu = batch.y_dict[NodeTypes.BUS][:, SolutionBusIndices.VOLTAGE_MAGNITUDE]

        abs_errors_pu = (magnitudes_pred_pu - magnitudes_target_pu).abs()

        if self.value_type == ValueTypes.ABSOLUTE:
            base_kV = batch.x_dict[NodeTypes.BUS][:, GridBusIndices.BASE_KV]
            return super().update(convert_unit(abs_errors_pu, base_kV, self.unit))

        if self.value_type == ValueTypes.RELATIVE:
            return super().update(calculate_relative_values(abs_errors_pu, magnitudes_target_pu))


class VoltageAngleError(AggregatorMetric):
    """
    A metric that measures absolute and relative errors of the bus voltage angle predictions.
    """

    is_differentiable: Optional[bool] = True
    higher_is_better: Optional[bool] = False
    full_state_update: bool = True

    def __init__(self, aggr: str, value_type: str, unit: str = "radian", **kwargs):
        super().__init__(aggr, **kwargs)
        self.value_type = value_type
        self.unit = unit

        if value_type not in list(ValueTypes):
            raise ValueError(
                f"Value type '{value_type}' is not supported. Expected one of {[vt for vt in ValueTypes]}."
            )

        if unit not in list(AngleUnits):
            raise ValueError(f"Angle unit '{unit}' is not supported. Expected one of {[au for au in AngleUnits]}.")

    def update(self, batch: HeteroData, predictions: dict[str, Tensor]):
        angle_pred_rad = predictions[NodeTypes.BUS][:, SolutionBusIndices.VOLTAGE_ANGLE]
        angle_target_rad = batch.y_dict[NodeTypes.BUS][:, SolutionBusIndices.VOLTAGE_ANGLE]

        abs_errors_rad = (angle_pred_rad - angle_target_rad).abs()

        if self.value_type == ValueTypes.ABSOLUTE:
            if self.unit == AngleUnits.RADIAN:
                return super().update(abs_errors_rad)

            if self.unit == AngleUnits.DEGREE:
                return super().update(torch.rad2deg(abs_errors_rad))

        if self.value_type == ValueTypes.RELATIVE:
            return super().update(calculate_relative_values(abs_errors_rad, angle_target_rad))


class VoltageMagnitude(AggregatorMetric):
    """
    A metric that measures the bus-level voltages in the grid.
    """

    is_differentiable: Optional[bool] = True
    full_state_update: bool = True

    def __init__(self, aggr: str, unit: str = "per-unit", **kwargs):
        super().__init__(aggr, **kwargs)
        self.unit = unit

    def update(self, batch: HeteroData, predictions: dict[str, Tensor]):
        magnitudes_pred_pu = predictions[NodeTypes.BUS][:, SolutionBusIndices.VOLTAGE_MAGNITUDE]

        base_kV = batch.x_dict[NodeTypes.BUS][:, GridBusIndices.BASE_KV]
        return super().update(convert_unit(magnitudes_pred_pu, base_kV, self.unit))


class VoltageAngle(AggregatorMetric):
    """
    A metric that measures absolute and relative errors of the bus voltage angle predictions.
    """

    is_differentiable: Optional[bool] = True
    higher_is_better: Optional[bool] = False
    full_state_update: bool = True

    def __init__(self, aggr: str, unit: str = "radian", **kwargs):
        super().__init__(aggr, **kwargs)
        self.unit = unit

        if unit not in list(AngleUnits):
            raise ValueError(f"Angle unit '{unit}' is not supported. Expected one of {[au for au in AngleUnits]}.")

    def update(self, batch: HeteroData, predictions: dict[str, Tensor]):
        angle_pred_rad = predictions[NodeTypes.BUS][:, SolutionBusIndices.VOLTAGE_ANGLE]

        if self.unit == AngleUnits.RADIAN:
            return super().update(angle_pred_rad)

        if self.unit == AngleUnits.DEGREE:
            return super().update(torch.rad2deg(angle_pred_rad))
