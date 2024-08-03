from typing import Dict, List, Tuple

from torch import LongTensor
from torch_geometric.data import HeteroData

from opf_dataset_utils.enumerations import EdgeTypes


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
