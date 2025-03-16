from typing import Dict, Tuple

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from opf_dataset_utils.enumerations import (
    EdgeIndexIndices,
    EdgeTypes,
    GridLoadIndices,
    GridTransformerIndices,
    NodeTypes,
    SolutionGeneratorIndices,
)
from opf_dataset_utils.utils import aggregate_bus_level, extract_branch_admittances
from opf_dataset_utils.voltage import get_voltage_angles, get_voltages_magnitudes


def get_generator_powers(predictions: dict[str, Tensor]) -> Tensor:
    """
    Returns the generator powers per generator.

    Parameters
    ----------
    predictions: dict[str, Tensor]
        Predictions dictionary.

    Returns
    -------
    Sg: Tensor
        Generator powers per generator.
    """
    Sg = (
        predictions[NodeTypes.GENERATOR][:, SolutionGeneratorIndices.ACTIVE_POWER]
        + 1j * predictions[NodeTypes.GENERATOR][:, SolutionGeneratorIndices.REACTIVE_POWER]
    )

    return Sg


def get_load_powers(data: HeteroData) -> Tensor:
    """
    Returns the load powers per load.

    Parameters
    ----------
    data: HeteroData
        OPFData.

    Returns
    -------
    Sd: Tensor
        Load powers per load.
    """
    Sd = (
        data.x_dict[NodeTypes.LOAD][:, GridLoadIndices.ACTIVE_POWER]
        + 1j * data.x_dict[NodeTypes.LOAD][:, GridLoadIndices.REACTIVE_POWER]
    )

    return Sd


def get_generator_powers_per_bus(data: HeteroData, predictions: dict[str, Tensor]) -> Tensor:
    """
    Returns the generator powers per bus.

    Parameters
    ----------
    data: HeteroData
        OPFData.
    predictions: dict[str, Tensor]
        Predictions dictionary.

    Returns
    -------
    Sg_bus: Tensor
        Generator powers per bus.
    """

    num_buses = data.x_dict[NodeTypes.BUS].shape[0]

    Sg_bus = aggregate_bus_level(
        num_buses,
        index=data.edge_index_dict[(NodeTypes.BUS, EdgeTypes.GENERATOR_LINK, NodeTypes.GENERATOR)][
            EdgeIndexIndices.FROM
        ],
        src=get_generator_powers(predictions),
    )

    return Sg_bus


def get_load_powers_per_bus(data: HeteroData) -> Tensor:
    """
    Returns the load powers per bus.

    Parameters
    ----------
    data: HeteroData
        OPFData.

    Returns
    -------
    Sd_bus: Tensor
        Load powers per bus.
    """
    num_buses = data.x_dict[NodeTypes.BUS].shape[0]

    Sd_bus = aggregate_bus_level(
        num_buses,
        index=data.edge_index_dict[(NodeTypes.BUS, EdgeTypes.LOAD_LINK, NodeTypes.LOAD)][EdgeIndexIndices.FROM],
        src=get_load_powers(data),
    )

    return Sd_bus


def calculate_bus_powers(data: HeteroData, predictions: dict[str, Tensor]) -> Tensor:
    """
    Returns the complex powers per bus.

    Parameters
    ----------
    data: HeteroData
        OPFData.
    predictions: dict[str, Tensor]
        Predictions dictionary.

    Returns
    -------
    S_bus: Tensor
        Complex powers per bus.
    """
    Sg_bus = get_generator_powers_per_bus(data, predictions)
    Sd_bus = get_load_powers_per_bus(data)
    return Sg_bus - Sd_bus


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
