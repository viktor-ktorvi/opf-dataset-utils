import torch
from torch_geometric.datasets import OPFDataset
from torch_geometric.loader import DataLoader

from opf_dataset_utils.costs import (
    calculate_costs_per_generator,
    calculate_costs_per_grid,
)


def main():
    """
    Calculate costs per generator and per grid.
    Returns
    -------

    """
    dataset = OPFDataset(
        "data", case_name="pglib_opf_case14_ieee", split="val", topological_perturbations=False, num_groups=1
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    data = next(iter(loader)).to(device)

    costs_per_grid = calculate_costs_per_grid(data, data.y_dict)

    print("Costs per grid [$/h]:")
    print(costs_per_grid)

    costs_per_generator = calculate_costs_per_generator(data, data.y_dict)

    print("Costs per generator [$/h]:")
    print(costs_per_generator)


if __name__ == "__main__":
    main()
