from typing import Dict

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from opf_dataset_utils.enumerations import (
    EdgeIndexIndices,
    EdgeTypes,
    GridShuntIndices,
    NodeTypes,
)
from opf_dataset_utils.power import calculate_branch_powers, calculate_bus_powers
from opf_dataset_utils.utils import aggregate_bus_level
from opf_dataset_utils.voltage import get_voltages_magnitudes


def calculate_power_flow_errors(data: HeteroData, predictions: Dict) -> Tensor:
    """
    Calculate the power flow errors.
    Parameters
    ----------
    data: HeteroData
        OPFData.
    predictions: Dict
        Prediction dictionary. Must contain 'bus' and 'generator'.

    Returns
    -------
    power_flow_errors: Tensor
        Per-bus power flow errors.
    """
    num_buses = data.x_dict[NodeTypes.BUS].shape[0]

    # generator, load and shunt power at the bus
    S_bus = calculate_bus_powers(data, predictions)

    Ysh = (
        data.x_dict[NodeTypes.SHUNT][:, GridShuntIndices.CONDUCTANCE]
        + 1j * data.x_dict[NodeTypes.SHUNT][:, GridShuntIndices.SUSCEPTANCE]
    )

    Ysh_bus = aggregate_bus_level(
        num_buses,
        index=data.edge_index_dict[(NodeTypes.BUS, EdgeTypes.SHUNT_LINK, NodeTypes.SHUNT)][EdgeIndexIndices.FROM],
        src=Ysh,
    )

    Vm = get_voltages_magnitudes(predictions)
    Ssh_bus = torch.conj(Ysh_bus) * Vm**2

    # AC line and transformer branch powers going in(to) and out(from) of the bus
    S_ac_from, S_ac_to = calculate_branch_powers(data, predictions, EdgeTypes.AC_LINE)
    S_t_from, S_t_to = calculate_branch_powers(data, predictions, EdgeTypes.TRANSFORMER)

    ac_line_edge_index = data.edge_index_dict[(NodeTypes.BUS, EdgeTypes.AC_LINE, NodeTypes.BUS)]
    S_ac_from_bus = aggregate_bus_level(num_buses, index=ac_line_edge_index[EdgeIndexIndices.FROM], src=S_ac_from)
    S_ac_to_bus = aggregate_bus_level(num_buses, index=ac_line_edge_index[EdgeIndexIndices.TO], src=S_ac_to)

    transformer_edge_index = data.edge_index_dict[(NodeTypes.BUS, EdgeTypes.TRANSFORMER, NodeTypes.BUS)]
    S_t_from_bus = aggregate_bus_level(num_buses, index=transformer_edge_index[EdgeIndexIndices.FROM], src=S_t_from)
    S_t_to_bus = aggregate_bus_level(num_buses, index=transformer_edge_index[EdgeIndexIndices.TO], src=S_t_to)

    # conservation of power/energy
    return S_bus - Ssh_bus - (S_ac_from_bus + S_ac_to_bus + S_t_from_bus + S_t_to_bus)
