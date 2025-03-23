from typing import Type, Union

import torch
from scipy.sparse import csr_matrix
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


def get_branch_type_indices(branch_type: str) -> Union[Type[GridACLineIndices], Type[GridTransformerIndices]]:
    """
    Return the corresponding enum class for the given branch type.

    Parameters
    ----------
    branch_type: str
        One of ['ac_line', 'transformer'].

    Returns
    -------
    branch_type_enum: Union[Type[GridACLineIndices], Type[GridTransformerIndices]]
        Corresponding branch type enum class.
    Raises
    ------
    ValueError:
        If the branch type isn't supported.
    """
    if branch_type == EdgeTypes.AC_LINE:
        return GridACLineIndices
    elif branch_type == EdgeTypes.TRANSFORMER:
        return GridTransformerIndices
    else:
        raise ValueError(
            f"Branch type '{branch_type}' is not supported. Expected one of ['{EdgeTypes.AC_LINE}', '{EdgeTypes.TRANSFORMER}']"
        )


def get_tap_ratios(data: HeteroData, branch_type: str) -> Tensor:
    """
    Get the tap ratios of lines.

    Parameters
    ----------
    data: HeteroData
        OPF data.
    branch_type: str
        Branch type.

    Returns
    -------
    T_ij: Tensor
        Complex tap ratio.

    Raises
    ------
    ValueError
        If the branch_type is not supported.
    """
    if branch_type == EdgeTypes.TRANSFORMER:
        Tm_ij = data.edge_attr_dict[(NodeTypes.BUS, EdgeTypes.TRANSFORMER, NodeTypes.BUS)][
            :, GridTransformerIndices.TAP_MAGNITUDE
        ]
        T_phase = data.edge_attr_dict[(NodeTypes.BUS, EdgeTypes.TRANSFORMER, NodeTypes.BUS)][
            :, GridTransformerIndices.TAP_PHASE_SHIFT
        ]
        T_ij = Tm_ij * torch.exp(1j * T_phase)

        return T_ij

    if branch_type == EdgeTypes.AC_LINE:
        edge_index = data.edge_index_dict[(NodeTypes.BUS, EdgeTypes.AC_LINE, NodeTypes.BUS)]
        T_ij = torch.ones(edge_index.shape[1], dtype=torch.tensor(1j).dtype, device=edge_index.device)

        return T_ij

    raise ValueError(
        f"Branch type '{branch_type}' is not supported. Expected one of ['{EdgeTypes.AC_LINE}', '{EdgeTypes.TRANSFORMER}']"
    )


def csr_to_sparse_tensor(csr: csr_matrix, dtype: torch.dtype, size: tuple[int, ...]) -> Tensor:
    """
    Convert a scipy csr matrix to a torch csr tensor.

    Parameters
    ----------
    csr: csr_matrix
        csr matrix.
    dtype: torch.dtype
        Data type.
    size: tuple[int, ...]
        Size.

    Returns
    -------
    sparse_csr_tensor: Tensor
        Sparse csr tensor.
    """
    return torch.sparse_csr_tensor(
        crow_indices=torch.LongTensor(csr.indptr),
        col_indices=torch.LongTensor(csr.indices),
        values=torch.tensor(csr.data),
        dtype=dtype,
        size=size,
    )


def to_sparse_diag(tensor: Tensor):
    """
    Transform a 1D tensor into a sparse diagonal tensor.

    Parameters
    ----------
    tensor: Tensor
        1D tensor.

    Returns
    -------
    sparse_diag_tensor: Tensor
        A sparse diagonal tensor.
    """
    return torch.sparse.spdiags(tensor, torch.tensor(0), (len(tensor), len(tensor)), layout=torch.sparse_csr)
