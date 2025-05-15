import copy

import matplotlib.lines as mlines
import networkx as nx
import torch
from matplotlib.axes import Axes
from torch import LongTensor
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_networkx

from opf_dataset_utils.enumerations import (
    BusTypes,
    EdgeTypes,
    GridBusIndices,
    NodeTypes,
)
from opf_dataset_utils.plotting.utils import (
    display_legend,
    edge_index_to_list_of_tuples,
)


def draw_graph(
    heterogeneous_data: HeteroData,
    ax: Axes,
    show_legend: bool = True,
    highlight_nodes: dict[str, list[int]] = None,
    highlight_edges: dict[str, LongTensor] = None,
    **nx_node_kwargs,
):
    """
    Draw an OPF power grid graph.

    Parameters
    ----------
    heterogeneous_data: HeteroData
        heterogeneous OPFData.
    ax: Axes
        Axes.
    show_legend: bool
        To display the legend or not.
    highlight_nodes: dict[str, list[int]]
        Nodes to highlight.
    highlight_edges: dict[str, LongTensor]
        Highlight edge index.
    nx_node_kwargs
        draw_networkx_nodes kwargs.

    Returns
    -------
    """

    # to homogeneous and networkx
    homogenous_data = heterogeneous_data.to_homogeneous()
    G = to_networkx(homogenous_data)

    # extract node and edge types
    node_types_ids = dict(zip(heterogeneous_data.node_types, list(range(0, len(heterogeneous_data.node_types)))))
    edge_type_ids = dict(zip(heterogeneous_data.edge_types, list(range(0, len(heterogeneous_data.edge_types)))))

    # hardcode node colors and shape
    node_type_colors = {
        NodeTypes.BUS: "#00549f",
        NodeTypes.LOAD: "#8ebae5",
        NodeTypes.GENERATOR: "#f6a800",
        NodeTypes.SHUNT: "#9c9e9f",
    }

    node_type_shapes = {NodeTypes.BUS: "o", NodeTypes.LOAD: "v", NodeTypes.GENERATOR: "h", NodeTypes.SHUNT: "d"}

    # determine which node is the reference node
    reference_mask = heterogeneous_data.x_dict[NodeTypes.BUS][:, GridBusIndices.BUS_TYPE] == BusTypes.REFERENCE

    pos = nx.nx_pydot.graphviz_layout(G)

    # draw different edge types
    ac_line_mask = homogenous_data["edge_type"] == edge_type_ids[(NodeTypes.BUS, EdgeTypes.AC_LINE, NodeTypes.BUS)]
    transformer_mask = (
        homogenous_data["edge_type"] == edge_type_ids[(NodeTypes.BUS, EdgeTypes.TRANSFORMER, NodeTypes.BUS)]
    )

    equipment_link_mask = ~ac_line_mask * ~transformer_mask
    equipment_links = edge_index_to_list_of_tuples(
        homogenous_data.edge_index[:, equipment_link_mask]
    )  # same edge style for all the different equipment
    ac_lines = edge_index_to_list_of_tuples(homogenous_data.edge_index[:, ac_line_mask])
    transformers = edge_index_to_list_of_tuples(homogenous_data.edge_index[:, transformer_mask])

    # drawing edges with direction
    arrowsize = nx_node_kwargs.get("node_size", 300) // 15

    if len(equipment_links) > 0:
        line_collection = nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            arrows=True,
            edgelist=equipment_links,
            style=(0, (1, 3)),
            arrowsize=arrowsize,
        )

        # TODO check for existence of each edge type
        ax.add_artist(
            mlines.Line2D(
                [],
                [],
                color=line_collection[0].get_edgecolor(),
                linestyle=line_collection[0].get_linestyle(),
                label="equipment link",
            )
        )

    if len(ac_lines) > 0:
        line_collection = nx.draw_networkx_edges(
            G, pos, ax=ax, arrows=True, edgelist=ac_lines, style="solid", arrowsize=arrowsize
        )
        ax.add_artist(
            mlines.Line2D(
                [],
                [],
                color=line_collection[0].get_edgecolor(),
                linestyle=line_collection[0].get_linestyle(),
                label="AC line",
            )
        )

    if len(transformers) > 0:
        line_collection = nx.draw_networkx_edges(
            G, pos, ax=ax, arrows=True, edgelist=transformers, style=(0, (5, 5)), arrowsize=arrowsize
        )
        ax.add_artist(
            mlines.Line2D(
                [],
                [],
                color=line_collection[0].get_edgecolor(),
                linestyle=line_collection[0].get_linestyle(),
                label="transformer",
            )
        )

    # TODO to highlight node, first draw only that node but larger in a different color with alpha

    # draw different node types
    for node_type in node_types_ids.keys():
        node_mask = homogenous_data["node_type"] == node_types_ids[node_type]

        if highlight_nodes is not None and node_type in highlight_nodes:
            # TODO get the corresponding homogeneous node id
            highlight_node_kwargs = copy.deepcopy(nx_node_kwargs)

            node_size_multiplier = 3.5
            if "node_size" in highlight_node_kwargs:
                highlight_node_kwargs["node_size"] = highlight_node_kwargs["node_size"] * node_size_multiplier
            else:
                highlight_node_kwargs["node_size"] = 300 * node_size_multiplier

            nx.draw_networkx_nodes(
                G,
                pos,
                ax=ax,
                nodelist=torch.arange(0, homogenous_data.num_nodes)[node_mask][highlight_nodes[node_type]].tolist(),
                node_color="#fc03be",
                alpha=0.5,
                node_shape=node_type_shapes[node_type],
                label="highlighted node",
                **highlight_node_kwargs,
            )

        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            nodelist=torch.arange(0, homogenous_data.num_nodes)[node_mask].tolist(),
            node_color=node_type_colors[node_type],
            node_shape=node_type_shapes[node_type],
            label=node_type,
            **nx_node_kwargs,
        )

        # draw reference bus
        if node_type == NodeTypes.BUS:
            nx.draw_networkx_nodes(
                G,
                pos,
                ax=ax,
                nodelist=torch.arange(0, homogenous_data.num_nodes)[node_mask][reference_mask].tolist(),
                node_color="#cc071e",
                node_shape="s",
                label="reference bus",
                **nx_node_kwargs,
            )

    ax.axis("off")  # remove border

    if show_legend:
        display_legend(ax)
