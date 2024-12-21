from typing import Optional

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from opf_dataset_utils.costs import calculate_costs_per_generator, calculate_costs_per_grid
from opf_dataset_utils.physics.metrics.aggregation import AggregatorMetric


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

        mask = self.target_costs_per_grid.abs() > 0

        costs = self.costs_per_grid[mask]
        target_costs = self.target_costs_per_grid[mask]

        optimality_gap = (costs - target_costs) / target_costs
        optimality_gap = optimality_gap.abs() * 100.0

        super().update(optimality_gap)
