from typing import Tuple

import torch
from torch import LongTensor, Tensor
from torch_geometric.data import HeteroData

from opf_dataset_utils.enumerations import (
    EdgeTypes,
    GridACLineIndices,
    GridTransformerIndices,
    NodeTypes,
)


def aggregate_bus_level(num_buses: int, index: LongTensor, src: Tensor) -> Tensor:
    """
    Scatter sum operation to add each element at its corresponding bus.

    Parameters
    ----------
    num_buses: int
        Number of buses.
    index: LongTensor
        Indices controlling where each element gets summed (see torch.scatter_add_).
    src: Tensor
        Elements to be aggregated.
    Returns
    -------
    aggregated_values: Tensor
        Values aggregated to the buses (of the size num_buses).
    """
    return torch.zeros(num_buses, dtype=src.dtype, device=src.device).scatter_add_(dim=0, index=index, src=src)


def calculate_admittances(r: Tensor, x: Tensor) -> Tensor:
    """
    Calculate admittances from series parameters.
    Parameters
    ----------
    r: Tensor
        Series resistances.
    x: Tensor
        Series reactances.

    Returns
    -------
    admittances: Tensor
        Admittances.
    """
    sum_of_squares = r**2 + x**2
    return r / sum_of_squares - 1j * x / sum_of_squares


def extract_branch_admittances(data: HeteroData, branch_type: str) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Extract the series and charging admittances.
    Parameters
    ----------
    data: HeteroData
        OPFData.
    branch_type: str
        One of ['ac_line', 'transformer'].

    Returns
    -------
    Y_ij: Tensor
        Series admittances.
    Yc_ij: Tensor
        Charging admittance in the 'from' direction.
    Yc_ji
        Charging admittance in the 'to' direction.
    """
    if branch_type == EdgeTypes.AC_LINE:
        indices = GridACLineIndices
    elif branch_type == EdgeTypes.TRANSFORMER:
        indices = GridTransformerIndices
    else:
        raise ValueError(
            f"Branch type '{branch_type}' is not supported. Expected one of ['{EdgeTypes.AC_LINE}', '{EdgeTypes.TRANSFORMER}']"
        )

    Y_ij = calculate_admittances(
        data.edge_attr_dict[(NodeTypes.BUS, branch_type, NodeTypes.BUS)][:, indices.SERIES_RESISTANCE],
        data.edge_attr_dict[(NodeTypes.BUS, branch_type, NodeTypes.BUS)][:, indices.SERIES_REACTANCE],
    )

    Yc_ij = 1j * data.edge_attr_dict[(NodeTypes.BUS, branch_type, NodeTypes.BUS)][:, indices.CHARGING_SUSCEPTANCE_FROM]
    Yc_ji = 1j * data.edge_attr_dict[(NodeTypes.BUS, branch_type, NodeTypes.BUS)][:, indices.CHARGING_SUSCEPTANCE_TO]

    return Y_ij, Yc_ij, Yc_ji
