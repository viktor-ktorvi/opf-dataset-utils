from typing import Dict, List, Tuple

from torch import LongTensor
from torch_geometric.data import HeteroData

from opf_dataset_utils.enumerations import EdgeTypes


def set_equipment_node_distances(heterogenous_data: HeteroData, edge_type_ids: Dict, distance: float = 0.2) -> Dict:
    """
    Set the distances of the equipment nodes from their respective bus node.
    Parameters
    ----------
    heterogenous_data: HeteroData
        Heterogenous OPFData.
    edge_type_ids: Dict
        Dictionary of edge types and their corresponding IDs in the edge store.
    distance: float
        Distance setpoint.

    Returns
    -------
    node_distances: Dict
        Node distances.
    """
    edge_lengths = {
        EdgeTypes.GENERATOR_LINK: distance,
        EdgeTypes.LOAD_LINK: distance,
        EdgeTypes.SHUNT_LINK: distance,
    }

    node_distances = {}
    for edge_type in edge_type_ids.keys():
        if edge_type[1] in [EdgeTypes.AC_LINE, EdgeTypes.TRANSFORMER]:
            continue

        edge_index = heterogenous_data.edge_stores[edge_type_ids[edge_type]]["edge_index"]

        for i, j in zip(edge_index[0], edge_index[1]):
            source = i.item() + heterogenous_data.node_offsets[edge_type[0]]
            target = j.item() + heterogenous_data.node_offsets[edge_type[2]]
            if source not in node_distances:
                node_distances[source] = {}
            node_distances[source][target] = edge_lengths[edge_type[1]]

    return node_distances


def edge_index_to_list_of_tuples(edge_index: LongTensor) -> List[Tuple[int]]:
    """
    Turn an edge index into a list of tuples.
    Parameters
    ----------
    edge_index: LongTensor
        Edge index.

    Returns
    -------
    list_of_tuples: List[Tuple[int]]
        Edge index as a list of tuples.
    """
    edge_list = edge_index.tolist()
    return list(zip(edge_list[0], edge_list[1]))
