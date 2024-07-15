from typing import Dict, Tuple

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from opf_dataset_utils.enumerations import (
    EdgeIndexIndices,
    EdgeTypes,
    GridTransformerIndices,
    NodeTypes,
)
from opf_dataset_utils.physics.utils import extract_branch_admittances
from opf_dataset_utils.physics.voltage import (
    get_voltage_angles,
    get_voltages_magnitudes,
)


def calculate_branch_powers(data: HeteroData, predictions: Dict, branch_type: str) -> Tuple[Tensor, Tensor]:
    """
    Calculate the branch powers of the given branch type in both the from and to directions.
    Parameters
    ----------
    data: HeteroData
        OPFData.
    predictions: Dict
        Prediction dictionary. Must contain 'bus'.
    branch_type: str
        One of ['ac_line', 'transformer'].

    Returns
    -------
    S_ij: Tensor
        Branch powers in the 'from' direction.
    S_ji: Tensor
        Branch powers in the 'to' direction.
    """
    Y_ij, Yc_ij, Yc_ji = extract_branch_admittances(data, branch_type)

    edge_index = data.edge_index_dict[(NodeTypes.BUS, branch_type, NodeTypes.BUS)]

    Vm = get_voltages_magnitudes(predictions)

    Vm_i = Vm[edge_index[EdgeIndexIndices.FROM]]
    Vm_j = Vm[edge_index[EdgeIndexIndices.TO]]

    Va = get_voltage_angles(predictions)

    V_i = Vm_i * torch.exp(1j * Va[edge_index[EdgeIndexIndices.FROM]])
    V_j = Vm_j * torch.exp(1j * Va[edge_index[EdgeIndexIndices.TO]])

    if branch_type == EdgeTypes.TRANSFORMER:
        Tm_ij = data.edge_attr_dict[(NodeTypes.BUS, EdgeTypes.TRANSFORMER, NodeTypes.BUS)][
            :, GridTransformerIndices.TAP_MAGNITUDE
        ]
        T_phase = data.edge_attr_dict[(NodeTypes.BUS, EdgeTypes.TRANSFORMER, NodeTypes.BUS)][
            :, GridTransformerIndices.TAP_PHASE_SHIFT
        ]
        T_ij = Tm_ij * torch.exp(1j * T_phase)
    elif branch_type == EdgeTypes.AC_LINE:
        Tm_ij = torch.ones_like(Y_ij.real)
        T_ij = torch.ones_like(Y_ij)
    else:
        raise ValueError(
            f"Branch type '{branch_type}' is not supported. Expected one of ['{EdgeTypes.AC_LINE}', '{EdgeTypes.TRANSFORMER}']"
        )

    S_ij = torch.conj(Y_ij + Yc_ij) * (Vm_i / Tm_ij) ** 2 - torch.conj(Y_ij * V_j) * V_i / T_ij
    S_ji = torch.conj(Y_ij + Yc_ji) * Vm_j**2 - torch.conj(Y_ij * V_i / T_ij) * V_j

    return S_ij, S_ji
