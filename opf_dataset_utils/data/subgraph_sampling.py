import copy
import math
import random
from typing import List, Dict

import torch
from torch import LongTensor
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected

from opf_dataset_utils.enumerations import NodeTypes, EdgeTypes, GridBusIndices, BusTypes


def get_neighborhood(source_nodes: List[int], edge_index: LongTensor) -> List[int]:
    """
    Get the 1-hop neighbourhood of the input source nodes.
    Parameters
    ----------
    source_nodes: List[int]
        Source nodes.
    edge_index: LongTensor
        Edge index.

    Returns
    -------
    neighborhood: List[int]
        The 1-hop neighbourhood of the input source nodes.
    """
    return edge_index[1, torch.isin(edge_index[0], torch.LongTensor(source_nodes))].tolist()


def get_set_difference(a: List, b: List) -> List:
    """
    Get the set difference of two list.
    The difference consists of the elements that are in list a, but not in list b.
    Parameters
    ----------
    a: List
        Set being subtracted from.
    b: List
        Set being subtracted.

    Returns
    -------
    difference: List
        Set difference.
    """
    return list(set(a) - set(b))


def sample_buses(data: HeteroData, num_hops: int, bus_retention_ratio: float = 1.0, equipment_retention_ratio: float = 1.0) -> Dict[str, LongTensor]:
    """
    Sample a subset of buses (and attached equipment) from the given OPFData object.
    Parameters
    ----------
    data: HeteroData
        OPFData.
    num_hops: int
        Number of hops to take.
    bus_retention_ratio: float
        Bus retention ratio - the ratio of buses to be kept at each hop.
    equipment_retention_ratio: float
        Equipment retention ration - the ratio of equipment nodes to be kept in the final subgraph.

    Returns
    -------
    subsets_dict: Dict[str, List[int]]
        A dictionary of the nodes included in the subgraph corresponding to a node type.

    Raises
    ------
    ValueError:
        If the num_hops is less than 1; or if bus_retention_ratio or are not in (0, 1].
    """

    # TODO it's slower than it should be. See why.

    if num_hops < 1:
        raise ValueError(f"Expected number of hops to be > 0, but got {num_hops} instead.")

    if not 0 < bus_retention_ratio <= 1:
        raise ValueError(f"Expected the bus retention ratio to be in (0, 1], but got {bus_retention_ratio} instead.")

    if not 0 < equipment_retention_ratio <= 1:
        raise ValueError(f"Expected the equipment retention ratio to be in (0, 1], but got {equipment_retention_ratio} instead.")

    edge_index_dict = ToUndirected()(data).edge_index_dict

    bus_edge_types = [(NodeTypes.BUS, EdgeTypes.AC_LINE, NodeTypes.BUS),
                      (NodeTypes.BUS, EdgeTypes.TRANSFORMER, NodeTypes.BUS)]

    bus_subset = [random.randint(0, data.x_dict[NodeTypes.BUS].shape[0] - 1)]  # TODO add option to start from a specific region

    neighborhood = copy.deepcopy(bus_subset)

    for i in range(num_hops):
        if len(neighborhood) == 0:
            break
        prev_bus_subset = copy.deepcopy(bus_subset)

        for edge_type in bus_edge_types:
            new_neighbors = get_neighborhood(neighborhood, edge_index_dict[edge_type])
            new_neighbors = get_set_difference(new_neighbors, bus_subset)  # remove previously visited nodes

            new_neighbors = random.sample(new_neighbors, k=math.ceil(len(new_neighbors) * bus_retention_ratio))

            bus_subset += new_neighbors

        neighborhood = get_set_difference(bus_subset, prev_bus_subset)

    equipment_subsets_dict = {}
    for edge_type in get_set_difference(data.edge_types, bus_edge_types):
        source_type = edge_type[0]
        destination_type = edge_type[2]

        if source_type != NodeTypes.BUS:
            continue

        new_neighbors = get_neighborhood(bus_subset, edge_index_dict[edge_type])
        new_neighbors = random.sample(new_neighbors, k=math.ceil(len(new_neighbors) * equipment_retention_ratio))

        if destination_type in equipment_subsets_dict:
            equipment_subsets_dict[destination_type] += new_neighbors
        else:
            equipment_subsets_dict[destination_type] = new_neighbors

    subsets_dict = {**{NodeTypes.BUS.value: bus_subset}, **equipment_subsets_dict}
    for node_type in subsets_dict:
        subsets_dict[node_type] = torch.LongTensor(subsets_dict[node_type])

    return subsets_dict


def contains_reference_bus(subset_dict: Dict[str, LongTensor], data: HeteroData) -> bool:
    """
    Return true if the bus subset contains a reference bus.
    Parameters
    ----------
    subset_dict: Dict[str, List[int]]
        A dictionary of the nodes included in the subgraph corresponding to a node type.
    data: HeteroData
        OPFData.

    Returns
    -------
    contains_reference_bus_flag: bool
        An indicator of if the given subset contains the reference bus.
    """
    reference_bus_mask = data.x_dict[NodeTypes.BUS][:, GridBusIndices.BUS_TYPE] == BusTypes.REFERENCE
    reference_bus_index = torch.argwhere(reference_bus_mask)[0].item()

    return reference_bus_index in subset_dict[NodeTypes.BUS]


def is_within_size_range(subset_dict: Dict[str, LongTensor], minimum: int = 1, maximum: int = math.inf):
    """
    Return true if the number of buses in the subset is within the given size range.
    Parameters
    ----------
    subset_dict: Dict[str, List[int]]
        A dictionary of the nodes included in the subgraph corresponding to a node type.
    minimum: int
        Minimum size.
    maximum: int
        Maximum size.

    Returns
    -------
    is_within_size_range_flag: bool
        An indicator of if the number of buses in the subset is within the given size range.
    """
    return minimum <= len(subset_dict[NodeTypes.BUS]) <= maximum
