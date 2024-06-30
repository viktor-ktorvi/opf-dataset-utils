import networkx as nx
import torch
from matplotlib.axes import Axes
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_networkx

from opf_dataset_utils.enumerations import BusTypes, EdgeTypes, NodeTypes
from opf_dataset_utils.plotting.utils import (
    edge_index_to_list_of_tuples,
    set_equipment_node_distances,
)


def draw_graph(heterogenous_data: HeteroData, ax: Axes, **nx_node_kwargs):
    """
    Draw an OPF power grid graph.
    Parameters
    ----------
    heterogenous_data: HeteroData
        Heterogenous OPFData.
    ax: Axes
        Axes.
    nx_node_kwargs
        draw_networkx_nodes kwargs.

    Returns
    -------

    """

    # TODO maybe draw the bus subgraph first (calculate pos just for those);
    #  then for each draw it"s loads/generators/shunts (calculate pos for that tree with the bus as a root)

    # to homogeneous and networkx
    homogenous_data = heterogenous_data.to_homogeneous()
    G = to_networkx(homogenous_data)

    # extract node and edge types
    node_types_ids = dict(zip(heterogenous_data.node_types, list(range(0, len(heterogenous_data.node_types)))))
    edge_type_ids = dict(zip(heterogenous_data.edge_types, list(range(0, len(heterogenous_data.edge_types)))))

    # hardcode node colors and shape
    node_type_colors = {
        NodeTypes.BUS: "#00549f",
        NodeTypes.LOAD: "#8ebae5",
        NodeTypes.GENERATOR: "#f6a800",
        NodeTypes.SHUNT: "#9c9e9f",
    }

    node_type_shapes = {NodeTypes.BUS: "o", NodeTypes.LOAD: "v", NodeTypes.GENERATOR: "h", NodeTypes.SHUNT: "d"}

    # determine which node is the reference node
    reference_mask = heterogenous_data.x_dict[NodeTypes.BUS][:, 1] == BusTypes.REFERENCE
    reference_node = (
        heterogenous_data.node_offsets[NodeTypes.BUS]
        + torch.arange(0, heterogenous_data.x_dict[NodeTypes.BUS].shape[0])[reference_mask]
    )
    if len(reference_node) > 1:
        reference_node = reference_node[0]

    # attempt a decently clean layout
    pos = nx.kamada_kawai_layout(
        G,
        dist=set_equipment_node_distances(
            heterogenous_data, edge_type_ids, distance=0.15
        ),  # set equipment close to their corresponding buses
        pos=nx.bfs_layout(G, start=reference_node.item()),  # init with breath first search layout
    )

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
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=False, edgelist=equipment_links, style=":", label="equipment link")
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=False, edgelist=ac_lines, label="AC line")
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=False, edgelist=transformers, style="--", label="transformer")

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
                # node_shape=node_type_shapes[node_type],
                node_shape="s",
                label="reference bus",
                **nx_node_kwargs,
            )

    ax.axis("off")  # remove border
    legend = ax.legend()

    # make sure markers fit into the legend box
    for handle in legend.legend_handles:
        handle._sizes = [150]
