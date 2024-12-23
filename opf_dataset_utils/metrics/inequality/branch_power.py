from typing import Optional

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from opf_dataset_utils.enumerations import (
    EdgeIndexIndices,
    EdgeTypes,
    GridACLineIndices,
    GridTransformerIndices,
    NodeTypes,
)
from opf_dataset_utils.errors.inequality.branch_power import (
    calculate_branch_power_errors_from,
    calculate_branch_power_errors_to,
)
from opf_dataset_utils.metrics.aggregation import AggregatorMetric
from opf_dataset_utils.metrics.relative_values import (
    ValueTypes,
    calculate_relative_values,
)
from opf_dataset_utils.metrics.units import convert_unit


# TODO test and docs
class BranchPowerInequalityError(AggregatorMetric):
    """
    A metric for branch apparent power inequality errors.
    Combines both the AC Line and Transformer branch values, as well as the 'from' and 'to' directions.
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
        errors_from_ac_pu = calculate_branch_power_errors_from(batch, predictions, EdgeTypes.AC_LINE)
        errors_to_ac_pu = calculate_branch_power_errors_to(batch, predictions, EdgeTypes.AC_LINE)

        errors_from_transformer_pu = calculate_branch_power_errors_from(batch, predictions, EdgeTypes.TRANSFORMER)
        errors_to_transformer_pu = calculate_branch_power_errors_to(batch, predictions, EdgeTypes.TRANSFORMER)

        errors_pu = torch.cat(
            (errors_from_ac_pu, errors_to_ac_pu, errors_from_transformer_pu, errors_to_transformer_pu)
        )

        if self.value_type == ValueTypes.ABSOLUTE:
            baseMVA_per_bus = batch.x[batch.batch_dict[NodeTypes.BUS]]

            baseMVA_ac_from = baseMVA_per_bus[
                batch.edge_index_dict[NodeTypes.BUS, EdgeTypes.AC_LINE, NodeTypes.BUS][EdgeIndexIndices.FROM]
            ]
            baseMVA_ac_to = baseMVA_per_bus[
                batch.edge_index_dict[NodeTypes.BUS, EdgeTypes.AC_LINE, NodeTypes.BUS][EdgeIndexIndices.TO]
            ]

            baseMVA_transformer_from = baseMVA_per_bus[
                batch.edge_index_dict[NodeTypes.BUS, EdgeTypes.TRANSFORMER, NodeTypes.BUS][EdgeIndexIndices.FROM]
            ]
            baseMVA_transformer_to = baseMVA_per_bus[
                batch.edge_index_dict[NodeTypes.BUS, EdgeTypes.TRANSFORMER, NodeTypes.BUS][EdgeIndexIndices.TO]
            ]

            baseMVA_per_branch = torch.cat(
                (baseMVA_ac_from, baseMVA_ac_to, baseMVA_transformer_from, baseMVA_transformer_to)
            )
            return super().update(convert_unit(errors_pu, baseMVA_per_branch, self.unit))

        if self.value_type == ValueTypes.RELATIVE:
            S_max_ac_from = batch.edge_attr_dict[NodeTypes.BUS, EdgeTypes.AC_LINE, NodeTypes.BUS][
                :, GridACLineIndices.LONG_TERM_RATING
            ]
            S_max_ac_to = batch.edge_attr_dict[NodeTypes.BUS, EdgeTypes.AC_LINE, NodeTypes.BUS][
                :, GridACLineIndices.LONG_TERM_RATING
            ]

            S_max_transformer_from = batch.edge_attr_dict[NodeTypes.BUS, EdgeTypes.TRANSFORMER, NodeTypes.BUS][
                :, GridTransformerIndices.LONG_TERM_RATING
            ]
            S_max_transformer_to = batch.edge_attr_dict[NodeTypes.BUS, EdgeTypes.TRANSFORMER, NodeTypes.BUS][
                :, GridTransformerIndices.LONG_TERM_RATING
            ]

            S_max = torch.cat((S_max_ac_from, S_max_ac_to, S_max_transformer_from, S_max_transformer_to))
            return super().update(calculate_relative_values(errors_pu, S_max))
