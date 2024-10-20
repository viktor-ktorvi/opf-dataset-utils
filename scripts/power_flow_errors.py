import torch
from torch_geometric.datasets import OPFDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv, to_hetero

from opf_dataset_utils.physics.errors.power_flow import calculate_power_flow_errors


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

    mean_abs_errors_solution = calculate_power_flow_errors(batch, batch.y_dict).abs().mean()
    mean_abs_errors_untrained = calculate_power_flow_errors(batch, untrained_predictions).abs().mean()

    print("Mean power flow errors:")
    print(f"\tSolution: {mean_abs_errors_solution:.5e} [p.u.]")
    print(f"\tUntrained model prediction: {mean_abs_errors_untrained:.5f} [p.u.]")


if __name__ == "__main__":
    main()
