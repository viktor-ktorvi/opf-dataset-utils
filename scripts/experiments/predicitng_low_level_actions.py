import hydra
import matplotlib
import torch
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from torch_geometric.datasets import OPFDataset
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GraphSAGE, to_hetero
from torch_geometric.utils import to_undirected

from opf_dataset_utils import CONFIG_PATH
from opf_dataset_utils.enumerations import EdgeIndexIndices, EdgeTypes, NodeTypes
from opf_dataset_utils.plotting.draw import draw_graph


def set_rcParams():
    """
    Set rcParams.

    Returns
    -------
    """
    matplotlib.rcParams["figure.autolayout"] = True
    matplotlib.rcParams["figure.figsize"] = (15, 9)
    matplotlib.rcParams["figure.dpi"] = 130
    matplotlib.rcParams["font.size"] = 16


@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="experiments")
def main(cfg: DictConfig):
    """A basic way to choose a substation and to choose the buses inside the substation to act upon, both in a
    permutation equivariant way, I think."""
    set_rcParams()

    # load data
    dataset = OPFDataset(
        "data", case_name=cfg.data.case_name, split="val", topological_perturbations=False, num_groups=1
    )

    data = dataset[0]

    # preprocessing -- make edges undirected
    for edge_type in [
        (NodeTypes.BUS, EdgeTypes.AC_LINE, NodeTypes.BUS),
        (NodeTypes.BUS, EdgeTypes.TRANSFORMER, NodeTypes.BUS),
    ]:
        data[edge_type].edge_index, data[edge_type].edge_attr = to_undirected(
            data[edge_type].edge_index, data[edge_type].edge_attr
        )

    # define an example model
    gnn = to_hetero(
        GraphSAGE(
            in_channels=-1,  # lazy initialization
            hidden_channels=64,
            num_layers=2,
            out_channels=64,
        ),
        data.metadata(),
    )

    # get a hidden state
    h_dict = gnn(x=data.x_dict, edge_index=data.edge_index_dict)

    # somehow select the substation to act upon
    # for example
    substation_id = h_dict[NodeTypes.BUS].norm(dim=1).argmax().item()

    # get the local neighborhood of the selected substation
    loader = NeighborLoader(
        data, num_neighbors=[-1], shuffle=True, input_nodes=(NodeTypes.BUS, torch.tensor([substation_id])), batch_size=1
    )
    local_subgraph = next(iter(loader))

    # select the actions to take inside the selected substation
    # this could, maybe, be implemented in a nicer way, without the for loop over edge types, but the essence stays the same
    counter = 0
    where_did_the_edge_come_from = {}
    dot_product_list = []
    for edge_type in local_subgraph.edge_types:
        edge_index = local_subgraph.edge_index_dict[edge_type]
        if edge_index.shape[1] == 0:
            continue

        node_type_from = edge_type[0]
        node_type_to = edge_type[2]

        # this is the important part -- this is a simple, permutation equivariant way to do an edge-level prediction
        # it can be more complicated, e.g., edge features can be added,
        # but the core concept is to combine the hidden states of nodes on the opposing sides of an edge

        h_from = h_dict[node_type_from][edge_index[EdgeIndexIndices.FROM]]
        h_to = h_dict[node_type_to][edge_index[EdgeIndexIndices.TO]]

        dot_product = (h_from * h_to).sum(dim=-1)
        dot_product_list.append(dot_product)

        # this is just to know how to get the index of each edge back in the PyG data format later
        where_did_the_edge_come_from[edge_type] = torch.arange(counter, counter + len(dot_product))
        counter += len(dot_product)

    # this is a way to get a specific number of actions
    # I guess it'd be better if the number of actions was chosen dynamically
    # for example, using a threshold, idk.
    # the action here is, basically, -- which bus' state is going to be toggled
    top_k_args = torch.hstack(dot_product_list).topk(2).indices

    # this is just translating the chosen actions into the indices in the PyG data format
    highlight_action_target_nodes = {}
    for edge_type in where_did_the_edge_come_from:
        for arg in top_k_args:
            edge_index_arg = torch.where(where_did_the_edge_come_from[edge_type] == arg)[0]
            if edge_index_arg.shape[0] == 0:
                continue

            chosen_action_target_node = local_subgraph.edge_index_dict[edge_type][
                EdgeIndexIndices.FROM, edge_index_arg
            ].item()

            node_type = edge_type[0]
            if node_type not in highlight_action_target_nodes:
                highlight_action_target_nodes[node_type] = []

            highlight_action_target_nodes[node_type].append(chosen_action_target_node)

    # in the first plot, the chosen substation is highlighted
    fig, axs = plt.subplots(1, 2)
    axs[0].set_title("Original graph and the selected substation")
    draw_graph(data, ax=axs[0], node_size=300, highlight_nodes={NodeTypes.BUS: [substation_id]})

    # in the second plot the nodes whose connections should be acted upon should be highlighted
    # I didn't highlight the edges because I couldn't find an easy way to do that
    axs[1].set_title("Selected substation subgraph")
    draw_graph(local_subgraph, ax=axs[1], node_size=300, highlight_nodes=highlight_action_target_nodes)
    plt.show()


if __name__ == "__main__":
    main()
