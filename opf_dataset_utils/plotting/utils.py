from typing import List, Tuple

from matplotlib.axes import Axes
from torch import LongTensor


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


def display_legend(ax: Axes, **kwargs):
    """
    Display legend. Resize markers so that they fit the legend.

    Parameters
    ----------
    ax: Axes
        Axes.
    kwargs
        Legend kwargs.

    Returns
    -------
    """
    legend = ax.legend(**kwargs)

    # make sure markers fit into the legend box
    for handle in legend.legend_handles:
        handle._sizes = [150]
