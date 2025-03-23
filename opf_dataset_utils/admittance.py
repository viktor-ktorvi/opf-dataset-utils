from typing import Tuple

import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch import LongTensor, Tensor
from torch_geometric.data import HeteroData

from opf_dataset_utils.enumerations import (
    EdgeIndexIndices,
    EdgeTypes,
    GridShuntIndices,
    NodeTypes,
)
from opf_dataset_utils.utils import (
    aggregate_bus_level,
    csr_to_sparse_tensor,
    get_branch_type_indices,
    get_tap_ratios,
    to_sparse_diag,
)


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
    indices = get_branch_type_indices(branch_type)

    Y_ij = calculate_admittances(
        data.edge_attr_dict[(NodeTypes.BUS, branch_type, NodeTypes.BUS)][:, indices.SERIES_RESISTANCE],
        data.edge_attr_dict[(NodeTypes.BUS, branch_type, NodeTypes.BUS)][:, indices.SERIES_REACTANCE],
    )

    Yc_ij = 1j * data.edge_attr_dict[(NodeTypes.BUS, branch_type, NodeTypes.BUS)][:, indices.CHARGING_SUSCEPTANCE_FROM]
    Yc_ji = 1j * data.edge_attr_dict[(NodeTypes.BUS, branch_type, NodeTypes.BUS)][:, indices.CHARGING_SUSCEPTANCE_TO]

    return Y_ij, Yc_ij, Yc_ji


def calculate_branch_admittance_matrix_parameters(
    data: HeteroData, branch_type: str
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Calculate the branch admittance matrix parameters for the given branch type.

    Parameters
    ----------
    data: HeteroData
        OPF data.
    branch_type: str
        Branch type.

    Returns
    -------
    Y_ff: Tensor
    Y_ft: Tensor
    Y_tf: Tensor
    Y_tt: Tensor
    """
    Y_series, y_charging_from, y_charging_to = extract_branch_admittances(data, branch_type)

    tap_ratios = get_tap_ratios(data, branch_type)

    Y_ff = (Y_series + y_charging_from) / tap_ratios.abs() ** 2
    Y_ft = -Y_series / torch.conj(tap_ratios)
    Y_tf = -Y_series / tap_ratios
    Y_tt = Y_series + y_charging_to

    return Y_ff, Y_ft, Y_tf, Y_tt


def get_bus_shunt_admittance(data: HeteroData) -> Tensor:
    """
    Get the per-bus shunt admittances.

    Parameters
    ----------
    data: HeteroData
        OPF data.

    Returns
    -------
    Ysh_bus: Tensor
        Shunt admittances per bus.
    """
    Ysh = (
        data.x_dict[NodeTypes.SHUNT][:, GridShuntIndices.CONDUCTANCE]
        + 1j * data.x_dict[NodeTypes.SHUNT][:, GridShuntIndices.SUSCEPTANCE]
    )

    Ysh_bus = aggregate_bus_level(
        data.num_nodes_dict[NodeTypes.BUS],
        index=data.edge_index_dict[(NodeTypes.BUS, EdgeTypes.SHUNT_LINK, NodeTypes.SHUNT)][EdgeIndexIndices.FROM],
        src=Ysh,
    )

    return Ysh_bus


def get_connectivity_matrices(edge_index: LongTensor, num_nodes: int) -> tuple[Tensor, Tensor]:
    """

    Parameters
    ----------
    edge_index
    num_nodes

    Returns
    -------

    """

    num_edges = edge_index.shape[1]

    C_f = csr_matrix((np.ones(num_edges), (range(num_edges), edge_index[0])), (num_edges, num_nodes))
    C_t = csr_matrix((np.ones(num_edges), (range(num_edges), edge_index[1])), (num_edges, num_nodes))

    C_f = csr_to_sparse_tensor(C_f, dtype=torch.int64, size=C_f.shape)
    C_t = csr_to_sparse_tensor(C_t, dtype=torch.int64, size=C_f.shape)

    return C_f, C_t


def calculate_admittance_matrix(data: HeteroData) -> tuple[Tensor, Tensor, Tensor]:
    """
    Calculate the admittance matrix (https://matpower.org/docs/manual.pdf).
    Parameters
    ----------
    data: HeteroData
        OPF data.

    Returns
    -------
    Y_bus: Tensor
        Admittance matrix.
    Y_f: Tensor
        Branch admittance matrix in the 'from' direction.
    Y_t: Tensor
        Branch admittance matrix in the 'to' direction.
    """

    edge_index = LongTensor(
        torch.hstack(
            (
                data.edge_index_dict[(NodeTypes.BUS, EdgeTypes.AC_LINE, NodeTypes.BUS)],
                data.edge_index_dict[(NodeTypes.BUS, EdgeTypes.TRANSFORMER, NodeTypes.BUS)],
            )
        )
    )

    admittance_matrix_parts = []
    for admittance_pair in zip(
        calculate_branch_admittance_matrix_parameters(data, EdgeTypes.AC_LINE),
        calculate_branch_admittance_matrix_parameters(data, EdgeTypes.TRANSFORMER),
    ):
        admittance_matrix_parts.append(torch.hstack(admittance_pair))

    Y_ff, Y_ft, Y_tf, Y_tt = admittance_matrix_parts

    num_buses = data.num_nodes_dict[NodeTypes.BUS]
    C_f, C_t = get_connectivity_matrices(edge_index, num_buses)
    C_f = C_f.to(Y_ff.dtype)
    C_t = C_t.to(Y_ff.dtype)

    Y_f = to_sparse_diag(Y_ff) @ C_f + to_sparse_diag(Y_ft) @ C_t
    Y_t = to_sparse_diag(Y_tf) @ C_f + to_sparse_diag(Y_tt) @ C_t

    C_f_tr = C_f.transpose(0, 1)
    C_t_tr = C_t.transpose(0, 1)

    Y_sh_diag = to_sparse_diag(get_bus_shunt_admittance(data))

    Y_bus = C_f_tr @ Y_f + C_t_tr @ Y_t + Y_sh_diag

    return Y_bus, Y_f, Y_t
