from typing import Dict

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from opf_dataset_utils.enumerations import (
    EdgeIndexIndices,
    EdgeTypes,
    GridLoadIndices,
    GridShuntIndices,
    NodeTypes,
    SolutionBusIndices,
    SolutionGeneratorIndices,
)
from opf_dataset_utils.physics.power import calculate_branch_powers
from opf_dataset_utils.physics.utils import aggregate_bus_level


def calculate_power_flow_errors(data: HeteroData, targets: Dict) -> Tensor:
    """
    Calculate the power flow errors.
    Parameters
    ----------
    data: HeteroData
        OPFData.
    targets: Dict
        Target dictionary. Must contain 'bus' and 'generator'.

    Returns
    -------
    power_flow_errors: Tensor
        Per-bus power flow errors.
    """
    num_buses = data.x_dict[NodeTypes.BUS].shape[0]

    # generator, load and shunt power at the bus
    Sg = (
        targets[NodeTypes.GENERATOR][:, SolutionGeneratorIndices.ACTIVE_POWER]
        + 1j * targets[NodeTypes.GENERATOR][:, SolutionGeneratorIndices.REACTIVE_POWER]
    )
    Sd = (
        data.x_dict[NodeTypes.LOAD][:, GridLoadIndices.ACTIVE_POWER]
        + 1j * data.x_dict[NodeTypes.LOAD][:, GridLoadIndices.REACTIVE_POWER]
    )
    Ysh = (
        data.x_dict[NodeTypes.SHUNT][:, GridShuntIndices.CONDUCTANCE]
        + 1j * data.x_dict[NodeTypes.SHUNT][:, GridShuntIndices.SUSCEPTANCE]
    )

    Sg_bus = aggregate_bus_level(
        num_buses,
        index=data.edge_index_dict[(NodeTypes.BUS, EdgeTypes.GENERATOR_LINK, NodeTypes.GENERATOR)][
            EdgeIndexIndices.FROM
        ],
        src=Sg,
    )
    Sd_bus = aggregate_bus_level(
        num_buses,
        index=data.edge_index_dict[(NodeTypes.BUS, EdgeTypes.LOAD_LINK, NodeTypes.LOAD)][EdgeIndexIndices.FROM],
        src=Sd,
    )
    Ysh_bus = aggregate_bus_level(
        num_buses,
        index=data.edge_index_dict[(NodeTypes.BUS, EdgeTypes.SHUNT_LINK, NodeTypes.SHUNT)][EdgeIndexIndices.FROM],
        src=Ysh,
    )

    Vm = targets[NodeTypes.BUS][:, SolutionBusIndices.VOLTAGE_MAGNITUDE]
    Ssh_bus = torch.conj(Ysh_bus) * Vm**2

    # AC line and transformer branch powers going in(to) and out(from) of the bus
    S_ac_from, S_ac_to = calculate_branch_powers(data, targets, EdgeTypes.AC_LINE)
    S_t_from, S_t_to = calculate_branch_powers(data, targets, EdgeTypes.TRANSFORMER)

    ac_line_edge_index = data.edge_index_dict[(NodeTypes.BUS, EdgeTypes.AC_LINE, NodeTypes.BUS)]
    S_ac_from_bus = aggregate_bus_level(num_buses, index=ac_line_edge_index[EdgeIndexIndices.FROM], src=S_ac_from)
    S_ac_to_bus = aggregate_bus_level(num_buses, index=ac_line_edge_index[EdgeIndexIndices.TO], src=S_ac_to)

    transformer_edge_index = data.edge_index_dict[(NodeTypes.BUS, EdgeTypes.TRANSFORMER, NodeTypes.BUS)]
    S_t_from_bus = aggregate_bus_level(num_buses, index=transformer_edge_index[EdgeIndexIndices.FROM], src=S_t_from)
    S_t_to_bus = aggregate_bus_level(num_buses, index=transformer_edge_index[EdgeIndexIndices.TO], src=S_t_to)

    # conservation of power/energy
    return Sg_bus - Sd_bus - Ssh_bus - (S_ac_from_bus + S_ac_to_bus + S_t_from_bus + S_t_to_bus)
