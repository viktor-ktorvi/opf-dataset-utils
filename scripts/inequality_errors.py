from typing import Dict

import torch
from torch_geometric.data import HeteroData
from torch_geometric.datasets import OPFDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv, to_hetero

from opf_dataset_utils.enumerations import EdgeTypes
from opf_dataset_utils.errors.inequality.branch_power import (
    calculate_branch_power_errors_from,
    calculate_branch_power_errors_to,
)
from opf_dataset_utils.errors.inequality.generator_power import (
    calculate_lower_active_power_errors,
    calculate_lower_reactive_power_errors,
    calculate_upper_active_power_errors,
    calculate_upper_reactive_power_errors,
)
from opf_dataset_utils.errors.inequality.voltage import (
    calculate_lower_voltage_angle_difference_errors,
    calculate_lower_voltage_magnitude_errors,
    calculate_upper_voltage_angle_difference_errors,
    calculate_upper_voltage_magnitude_errors,
)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = GraphConv(-1, 16)
        self.conv2 = GraphConv(16, 2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()

        x = self.conv2(x, edge_index)
        return x


def print_max_errors(data: HeteroData, predictions: Dict):
    # voltage
    spacing = (30, 25, 25)
    print(
        f"{'Upper Vm:':{spacing[0]}} {calculate_upper_voltage_magnitude_errors(data, predictions).abs().max().item():{spacing[1]}} {'[p.u.]':{spacing[2]}}"
    )
    print(
        f"{'Lower Vm:':{spacing[0]}} {calculate_lower_voltage_magnitude_errors(data, predictions).abs().max().item():{spacing[1]}} {'[p.u.]':{spacing[2]}}"
    )

    print(
        f"{'Upper Va diff. (transformers):':{spacing[0]}} {calculate_upper_voltage_angle_difference_errors(data, predictions, EdgeTypes.TRANSFORMER).abs().max().item():{spacing[1]}} {'[rad]':{spacing[2]}}"
    )
    print(
        f"{'Upper Va diff. (AC lines):':{spacing[0]}} {calculate_upper_voltage_angle_difference_errors(data, predictions, EdgeTypes.AC_LINE).abs().max().item():{spacing[1]}} {'[rad]':{spacing[2]}}"
    )

    print(
        f"{'Lower Va diff. (transformers):':{spacing[0]}} {calculate_lower_voltage_angle_difference_errors(data, predictions, EdgeTypes.TRANSFORMER).abs().max().item():{spacing[1]}} {'[rad]':{spacing[2]}}"
    )
    print(
        f"{'Lower Va diff. (AC lines):':{spacing[0]}} {calculate_lower_voltage_angle_difference_errors(data, predictions, EdgeTypes.AC_LINE).abs().max().item():{spacing[1]}} {'[rad]':{spacing[2]}}"
    )

    # generator power
    print(
        f"{'Upper Pg:':{spacing[0]}} {calculate_upper_active_power_errors(data, predictions).abs().max().item():{spacing[1]}} {'[p.u.]':{spacing[2]}}"
    )
    print(
        f"{'Lower Pg:':{spacing[0]}} {calculate_lower_active_power_errors(data, predictions).abs().max().item():{spacing[1]}} {'[p.u.]':{spacing[2]}}"
    )

    print(
        f"{'Upper Qg:':{spacing[0]}} {calculate_upper_reactive_power_errors(data, predictions).abs().max().item():{spacing[1]}} {'[p.u.]':{spacing[2]}}"
    )
    print(
        f"{'Lower Qg:':{spacing[0]}} {calculate_lower_reactive_power_errors(data, predictions).abs().max().item():{spacing[1]}} {'[p.u.]':{spacing[2]}}"
    )

    # branch (apparent) power
    print(
        f"{'Upper S_ij (transformers):':{spacing[0]}} {calculate_branch_power_errors_from(data, predictions, EdgeTypes.TRANSFORMER).abs().max().item():{spacing[1]}} {'[p.u.]':{spacing[2]}}"
    )
    print(
        f"{'Upper S_ij (AC lines):':{spacing[0]}} {calculate_branch_power_errors_from(data, predictions, EdgeTypes.AC_LINE).abs().max().item():{spacing[1]}} {'[p.u.]':{spacing[2]}}"
    )

    print(
        f"{'Upper S_ji (transformers):':{spacing[0]}} {calculate_branch_power_errors_to(data, predictions, EdgeTypes.TRANSFORMER).abs().max().item():{spacing[1]}} {'[p.u.]':{spacing[2]}}"
    )
    print(
        f"{'Upper S_ji (AC lines):':{spacing[0]}} {calculate_branch_power_errors_to(data, predictions, EdgeTypes.AC_LINE).abs().max().item():{spacing[1]}} {'[p.u.]':{spacing[2]}}"
    )


def main():
    """
    Calculate the various inequality violations of the solution and an untrained model.

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
        predictions = untrained_model(batch.x_dict, batch.edge_index_dict)

    print("\nWorst case violations:\n")

    print("Solution:\n")
    print_max_errors(batch, batch.y_dict)

    print("\nUntrained model:\n")
    print_max_errors(batch, predictions)


if __name__ == "__main__":
    main()
