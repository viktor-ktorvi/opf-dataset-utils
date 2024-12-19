from enum import StrEnum

import torch
from torch import Tensor
from torchmetrics import Metric


class AggregationTypes(StrEnum):
    MIN = "min"
    MAX = "max"
    MEAN = "mean"


class AggregatorMetric(Metric):
    """
    A metric that aggregates values in a specifiable way.
    """

    def __init__(self, aggr: str, **kwargs):
        super().__init__(**kwargs)

        self.aggr = aggr

        if self.aggr == AggregationTypes.MIN:
            self.add_state("min_value", default=torch.tensor(torch.inf), dist_reduce_fx="sum")
            return

        if self.aggr == AggregationTypes.MAX:
            self.add_state("max_value", default=torch.tensor(-torch.inf), dist_reduce_fx="sum")
            return

        if self.aggr == AggregationTypes.MEAN:
            self.add_state("value_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("num_values", default=torch.tensor(0), dist_reduce_fx="sum")
            return

        raise ValueError(f"Aggregation type '{aggr}' is not supported. Expected one of {[str(a) for a in AggregationTypes]}")

    def update(self, values: Tensor):
        if self.aggr == AggregationTypes.MIN:
            if values.min() < self.min_value:
                self.min_value = values.min()

        if self.aggr == AggregationTypes.MAX:
            if values.max() > self.max_value:
                self.max_value = values.max()

        if self.aggr == AggregationTypes.MEAN:
            self.value_sum += values.sum()
            self.num_values += values.numel()

    def compute(self) -> Tensor:
        if self.aggr == AggregationTypes.MIN:
            return self.min_value

        if self.aggr == AggregationTypes.MAX:
            return self.max_value

        if self.aggr == AggregationTypes.MEAN:
            return self.value_sum / self.num_values
