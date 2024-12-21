from typing import Optional

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from opf_dataset_utils.costs import calculate_costs_per_generator
from opf_dataset_utils.physics.metrics.aggregation import AggregatorMetric


class OptimalityGap(AggregatorMetric):
    """
    A metric of the optimality hap between the per-generator costs of the predictions and the targets expressed in percentage points %.
    """

    is_differentiable: Optional[bool] = True
    full_state_update: bool = True

    costs_per_generator: Tensor
    target_costs_per_generator: Tensor

    def __init__(self, aggr: str, **kwargs):
        super().__init__(aggr, **kwargs)
        self.add_state("costs_per_generator", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("target_costs_per_generator", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, batch: HeteroData, predictions: dict[str, Tensor]):
        self.costs_per_generator = calculate_costs_per_generator(batch, predictions)
        self.target_costs_per_generator = calculate_costs_per_generator(batch, batch.y_dict)

        optimality_gap = (self.costs_per_generator - self.target_costs_per_generator) / self.target_costs_per_generator
        optimality_gap = optimality_gap.abs() * 100.0

        super().update(optimality_gap)
