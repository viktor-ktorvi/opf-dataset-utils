import torch
from torch_geometric.datasets import OPFDataset
from torch_geometric.loader import DataLoader

from opf_dataset_utils.enumerations import EdgeTypes
from opf_dataset_utils.physics.power import calculate_branch_powers


def main():
    """
    Calculate branch powers.
    Returns
    -------

    """
    dataset = OPFDataset("data", case_name="pglib_opf_case14_ieee", split="val", topological_perturbations=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    batch = next(iter(loader)).to(device)

    ac_line_powers_from, ac_line_powers_to = calculate_branch_powers(batch, batch.y_dict, EdgeTypes.AC_LINE)
    transformer_powers_from, transformer_powers_to = calculate_branch_powers(batch, batch.y_dict, EdgeTypes.TRANSFORMER)

    print("AC line power flows [p.u.]:")
    print(ac_line_powers_from)  # TODO link in readme; add in scripts readme


if __name__ == "__main__":
    main()
