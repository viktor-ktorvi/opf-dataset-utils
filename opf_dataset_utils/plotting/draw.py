import networkx as nx
import torch
from matplotlib.axes import Axes
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


def draw_graph(heterogeneous_data: HeteroData, ax: Axes, show_legend: bool = True, **nx_node_kwargs):
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

    equipment_links = edge_index_to_list_of_tuples(
        homogenous_data.edge_index[:, ~ac_line_mask * ~transformer_mask]
    )  # same edge style for all the different equipment
    ac_lines = edge_index_to_list_of_tuples(homogenous_data.edge_index[:, ac_line_mask])
    transformers = edge_index_to_list_of_tuples(homogenous_data.edge_index[:, transformer_mask])

    # TODO drawing arrows doesn't show labels in legend
    nx.draw_networkx_edges(
        G, pos, ax=ax, arrows=False, edgelist=equipment_links, style="dotted", label="equipment link"
    )
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=False, edgelist=ac_lines, label="AC line")
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=False, edgelist=transformers, style=(0, (5, 1)), label="transformer")

    # draw different node types
    for node_type in node_types_ids.keys():
        node_mask = homogenous_data["node_type"] == node_types_ids[node_type]
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
