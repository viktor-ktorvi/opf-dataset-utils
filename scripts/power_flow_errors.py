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
    dataset = OPFDataset("data", case_name="pglib_opf_case14_ieee", split="val", topological_perturbations=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    batch = next(iter(loader)).to(device)

    untrained_model = to_hetero(Model(), batch.metadata())
    untrained_model.to(device)

    with torch.no_grad():
        predictions = untrained_model(batch.x_dict, batch.edge_index_dict)

    print(batch)
    print("Mean power flow errors:")
    print(f"\tSolution: {calculate_power_flow_errors(batch, batch.y_dict).abs().mean():.5e}")
    print(f"\tUntrained model prediction: {calculate_power_flow_errors(batch, predictions).abs().mean():.5f}")


if __name__ == "__main__":
    main()
