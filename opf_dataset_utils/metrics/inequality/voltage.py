from enum import StrEnum
from typing import Optional

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from opf_dataset_utils.enumerations import (
    EdgeTypes,
    GridACLineIndices,
    GridBusIndices,
    GridTransformerIndices,
    NodeTypes,
)
from opf_dataset_utils.errors.inequality.voltage import (
    calculate_lower_voltage_angle_difference_errors,
    calculate_lower_voltage_magnitude_errors,
    calculate_upper_voltage_angle_difference_errors,
    calculate_upper_voltage_magnitude_errors,
)
from opf_dataset_utils.metrics.aggregation import AggregatorMetric
from opf_dataset_utils.metrics.inequality.bound_types import BoundTypes
from opf_dataset_utils.metrics.relative_values import (
    ValueTypes,
    calculate_relative_values,
)
from opf_dataset_utils.metrics.units import convert_unit


# TODO test
#  detailed docs
class VoltageMagnitudeInequalityError(AggregatorMetric):
    """
    A metric for voltage magnitude inequality violations.
    """

    is_differentiable: Optional[bool] = True
    higher_is_better: Optional[bool] = False
    full_state_update: bool = True

    def __init__(self, aggr: str, bound_type: str, value_type: str, unit: str = "per-unit", **kwargs):
        super().__init__(aggr, **kwargs)
        self.unit = unit
        self.value_type = value_type
        self.bound_type = bound_type

        if value_type not in list(ValueTypes):
            raise ValueError(
                f"Value type '{value_type}' is not supported. Expected one of {[vt for vt in ValueTypes]}."
            )

        if bound_type not in list(BoundTypes):
            raise ValueError(
                f"Bound type '{bound_type}' is not supported. Expected one of {[bt for bt in BoundTypes]}."
            )

    def update(self, batch: HeteroData, predictions: dict[str, Tensor]):
        errors_pu = None
        if self.bound_type == BoundTypes.UPPER:
            errors_pu = calculate_upper_voltage_magnitude_errors(batch, predictions).abs()

        if self.bound_type == BoundTypes.LOWER:
            errors_pu = calculate_lower_voltage_magnitude_errors(batch, predictions).abs()

        if self.value_type == ValueTypes.ABSOLUTE:
            base_kV = batch.x_dict[NodeTypes.BUS][:, GridBusIndices.BASE_KV]
            return super().update(convert_unit(errors_pu, base_kV, self.unit))

        if self.value_type == ValueTypes.RELATIVE:
            Vm_max = batch.x_dict[NodeTypes.BUS][:, GridBusIndices.VOLTAGE_MAX]
            Vm_min = batch.x_dict[NodeTypes.BUS][:, GridBusIndices.VOLTAGE_MIN]

            return super().update(calculate_relative_values(errors_pu, Vm_max - Vm_min))


class AngleUnits(StrEnum):
    RADIAN = "radian"
    DEGREE = "degree"


class VoltageAngleDifferenceInequalityError(AggregatorMetric):
    """
    A metric for voltage angle difference inequality violations.
    """

    is_differentiable: Optional[bool] = True
    higher_is_better: Optional[bool] = False
    full_state_update: bool = True

    def __init__(self, aggr: str, bound_type: str, value_type: str, unit: str = "radian", **kwargs):
        super().__init__(aggr, **kwargs)
        self.value_type = value_type
        self.bound_type = bound_type
        self.unit = unit

        if value_type not in list(ValueTypes):
            raise ValueError(
                f"Value type '{value_type}' is not supported. Expected one of {[vt for vt in ValueTypes]}."
            )

        if bound_type not in list(BoundTypes):
            raise ValueError(
                f"Bound type '{bound_type}' is not supported. Expected one of {[bt for bt in BoundTypes]}."
            )

        if unit not in list(AngleUnits):
            raise ValueError(f"Angle unit '{unit}' is not supported. Expected one of {[au for au in AngleUnits]}.")

    def update(self, batch: HeteroData, predictions: dict[str, Tensor]):
        errors_rad = None
        if self.bound_type == BoundTypes.UPPER:
            errors_ac_line_rad = calculate_upper_voltage_angle_difference_errors(batch, predictions, EdgeTypes.AC_LINE)
            errors_transformer_rad = calculate_upper_voltage_angle_difference_errors(
                batch, predictions, EdgeTypes.TRANSFORMER
            )
            errors_rad = torch.cat((errors_ac_line_rad, errors_transformer_rad)).abs()

        if self.bound_type == BoundTypes.LOWER:
            errors_ac_line_rad = calculate_lower_voltage_angle_difference_errors(batch, predictions, EdgeTypes.AC_LINE)
            errors_transformer_rad = calculate_lower_voltage_angle_difference_errors(
                batch, predictions, EdgeTypes.TRANSFORMER
            )
            errors_rad = torch.cat((errors_ac_line_rad, errors_transformer_rad)).abs()

        if self.value_type == ValueTypes.ABSOLUTE:
            if self.unit == AngleUnits.RADIAN:
                return super().update(errors_rad)

            if self.unit == AngleUnits.DEGREE:
                return super().update(torch.rad2deg(errors_rad))

        if self.value_type == ValueTypes.RELATIVE:
            Va_diff_ac_line_max = batch.edge_attr_dict[NodeTypes.BUS, EdgeTypes.AC_LINE, NodeTypes.BUS][
                :, GridACLineIndices.ANGLE_DIFF_MAX
            ]
            Va_diff_transformer_max = batch.edge_attr_dict[NodeTypes.BUS, EdgeTypes.TRANSFORMER, NodeTypes.BUS][
                :, GridTransformerIndices.ANGLE_DIFF_MAX
            ]

            Va_diff_ac_line_min = batch.edge_attr_dict[NodeTypes.BUS, EdgeTypes.AC_LINE, NodeTypes.BUS][
                :, GridACLineIndices.ANGLE_DIFF_MIN
            ]
            Va_diff_transformer_min = batch.edge_attr_dict[NodeTypes.BUS, EdgeTypes.TRANSFORMER, NodeTypes.BUS][
                :, GridTransformerIndices.ANGLE_DIFF_MIN
            ]

            range_rad = torch.cat(
                (Va_diff_ac_line_max - Va_diff_ac_line_min, Va_diff_transformer_max - Va_diff_transformer_min)
            )

            return super().update(calculate_relative_values(errors_rad, range_rad))
