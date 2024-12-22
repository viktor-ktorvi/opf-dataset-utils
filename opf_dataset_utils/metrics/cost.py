from typing import Optional

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from opf_dataset_utils.costs import calculate_costs_per_grid
from opf_dataset_utils.metrics.aggregation import AggregatorMetric
from opf_dataset_utils.metrics.relative_values import calculate_relative_values


class OptimalityGap(AggregatorMetric):
    """
    A metric of the optimality hap between the per-grid costs of the predictions
    and the targets expressed in percentage points % -- basically a relative cost error.
    Excludes grids whose target cost is 0.
    """

    higher_is_better: Optional[bool] = False
    is_differentiable: Optional[bool] = True
    full_state_update: bool = True

    costs_per_grid: Tensor
    target_costs_per_grid: Tensor

    def __init__(self, aggr: str, **kwargs):
        super().__init__(aggr, **kwargs)
        self.add_state("costs_per_grid", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("target_costs_per_grid", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, batch: HeteroData, predictions: dict[str, Tensor]):
        self.costs_per_grid = calculate_costs_per_grid(batch, predictions)
        self.target_costs_per_grid = calculate_costs_per_grid(batch, batch.y_dict)

        optimality_gap = calculate_relative_values(
            self.costs_per_grid - self.target_costs_per_grid, self.target_costs_per_grid
        )

        super().update(optimality_gap)
