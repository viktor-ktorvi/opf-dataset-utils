import torch
from torch_geometric.datasets import OPFDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv, to_hetero
from torchmetrics import MetricCollection

from opf_dataset_utils.metrics.aggregation import AggregationTypes
from opf_dataset_utils.metrics.cost import OptimalityGap
from opf_dataset_utils.metrics.inequality.bound_types import BoundTypes
from opf_dataset_utils.metrics.inequality.branch_power import BranchPowerInequalityError
from opf_dataset_utils.metrics.inequality.generator_power import (
    GeneratorPowerInequalityError,
)
from opf_dataset_utils.metrics.inequality.voltage import (
    VoltageAngleDifferenceInequalityError,
    VoltageMagnitudeInequalityError,
)
from opf_dataset_utils.metrics.power import Power, PowerTypes
from opf_dataset_utils.metrics.power_flow import PowerFlowError
from opf_dataset_utils.metrics.variable.generator_power import GeneratorPowerError
from opf_dataset_utils.metrics.variable.voltage import (
    VoltageAngleError,
    VoltageMagnitudeError,
)


def create_opf_metrics(split: str) -> MetricCollection:
    metric_dict = {}
    for aggr in AggregationTypes:
        metric_dict[f"{split}/{aggr} optimality gap [%]"] = OptimalityGap(aggr=aggr)
        metric_dict[f"{split}/{aggr} absolute branch power inequality error [kVA]"] = BranchPowerInequalityError(
            aggr=aggr, value_type="absolute", unit="kilo"
        )
        metric_dict[f"{split}/{aggr} relative branch power inequality error [%]"] = BranchPowerInequalityError(
            aggr=aggr, value_type="relative"
        )

        metric_dict[f"{split}/{aggr} absolute voltage magnitude error [per-unit]"] = VoltageMagnitudeError(
            aggr=aggr, value_type="absolute"
        )
        metric_dict[f"{split}/{aggr} relative voltage magnitude error [%]"] = VoltageMagnitudeError(
            aggr=aggr, value_type="relative"
        )

        metric_dict[f"{split}/{aggr} absolute voltage angle error [deg]"] = VoltageAngleError(
            aggr=aggr, value_type="absolute", unit="degree"
        )
        metric_dict[f"{split}/{aggr} relative voltage angle error [%]"] = VoltageAngleError(
            aggr=aggr, value_type="relative"
        )

        for power_type in [PowerTypes.ACTIVE, PowerTypes.REACTIVE]:
            metric_dict[f"{split}/{aggr} absolute {power_type} generator power error [kVA]"] = GeneratorPowerError(
                aggr=aggr, power_type=power_type, unit="kilo", value_type="absolute"
            )

            metric_dict[f"{split}/{aggr} relative {power_type} generator power error [%]"] = GeneratorPowerError(
                aggr=aggr, power_type=power_type, value_type="relative"
            )

        for power_type in PowerTypes:
            metric_dict[f"{split}/{aggr} absolute {power_type} power flow error [kVA]"] = PowerFlowError(
                aggr=aggr, power_type=power_type, unit="kilo", value_type="absolute"
            )
            metric_dict[f"{split}/{aggr} relative {power_type} power flow error [%]"] = PowerFlowError(
                aggr=aggr, power_type=power_type, value_type="relative"
            )
            metric_dict[f"{split}/{aggr} {power_type} power [kVA]"] = Power(
                aggr=aggr, power_type=power_type, unit="kilo"
            )

        for bound_type in BoundTypes:
            metric_dict[
                f"{split}/{aggr} absolute {bound_type} voltage magnitude inequality error [per-unit]"
            ] = VoltageMagnitudeInequalityError(aggr=aggr, bound_type=bound_type, value_type="absolute")
            metric_dict[
                f"{split}/{aggr} relative {bound_type} voltage magnitude inequality error [%]"
            ] = VoltageMagnitudeInequalityError(aggr=aggr, bound_type=bound_type, value_type="relative")

            metric_dict[
                f"{split}/{aggr} absolute {bound_type} voltage angle difference error [deg]"
            ] = VoltageAngleDifferenceInequalityError(
                aggr=aggr, bound_type=bound_type, value_type="absolute", unit="degree"
            )
            metric_dict[
                f"{split}/{aggr} relative {bound_type} voltage angle difference error [%]"
            ] = VoltageAngleDifferenceInequalityError(aggr=aggr, bound_type=bound_type, value_type="relative")

            for power_type in [PowerTypes.ACTIVE, PowerTypes.REACTIVE]:
                metric_dict[
                    f"{split}/{aggr} absolute {bound_type} {power_type} generator power inequality error [kVA]"
                ] = GeneratorPowerInequalityError(
                    aggr=aggr, bound_type=bound_type, value_type="absolute", power_type=power_type, unit="kilo"
                )
                metric_dict[
                    f"{split}/{aggr} relative {bound_type} {power_type} generator power inequality error [%]"
                ] = GeneratorPowerInequalityError(
                    aggr=aggr, bound_type=bound_type, value_type="relative", power_type=power_type
                )

    return MetricCollection(metric_dict)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = GraphConv(-1, 16)
        self.conv2 = GraphConv(16, 2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()

        x = self.conv2(x, edge_index)
        return x


def main():
    """
    Calculate the power flow errors of the solution and an untrained model.
    Returns
    -------

    """
    dataset = OPFDataset(
        "data", case_name="pglib_opf_case14_ieee", split="val", topological_perturbations=False, num_groups=1
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    batch = next(iter(loader)).to(device)

    untrained_model = to_hetero(Model(), batch.metadata())
    untrained_model.to(device)

    with torch.no_grad():
        untrained_predictions = untrained_model(batch.x_dict, batch.edge_index_dict)

    metrics = create_opf_metrics("val").to(device)

    metrics(batch, untrained_predictions)

    metric_values = metrics.compute()

    for name, value in metric_values.items():
        print(f"\t{name:<75}: {value:>.5f}")


if __name__ == "__main__":
    main()
