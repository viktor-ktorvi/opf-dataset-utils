from enum import Enum

import hydra
import matplotlib
import networkx as nx
import numpy as np
import numpy.typing as npt
import pymetis
import torch
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from torch_geometric.data import Data, HeteroData
from torch_geometric.datasets import OPFDataset
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_networkx, to_undirected

from opf_dataset_utils import CONFIG_PATH
from opf_dataset_utils.enumerations import EdgeTypes, NodeTypes


def get_adjacency_list(graph: nx.Graph) -> list[npt.ArrayLike]:
    """
    Get adjacency list from a graph.

    Parameters
    ----------
    graph: nx.Graph
        Graph

    Returns
    -------
    adjacency_list: list[npt.ArrayLike]
        A list for each node in a graph containing an array of the nodes it's connected to.
    """
    adjacency_list = []
    for _, neighbors_dict in graph.adjacency():
        adjacency_list.append(np.array(list(neighbors_dict.keys())))

    return adjacency_list


class Virtual(str, Enum):
    NODE = "virtual_node"
    EDGE = "virtual_edge"


class ClusterVirtualNodes(BaseTransform):
    """Create virtual nodes using a graph clustering algorithm -- one virtual node per cluster."""

    num_clusters: int

    def __init__(self, num_clusters: int = 1):
        super().__init__()

        self.num_clusters = num_clusters

    def forward(self, data: HeteroData) -> HeteroData:
        if self.num_clusters < 1:
            return data

        edge_index = torch.concatenate(
            (
                data.edge_index_dict[NodeTypes.BUS, EdgeTypes.AC_LINE, NodeTypes.BUS],
                data.edge_index_dict[NodeTypes.BUS, EdgeTypes.TRANSFORMER, NodeTypes.BUS],
            ),
            dim=1,
        )
        edge_index = to_undirected(edge_index)

        num_buses = data.num_nodes_dict[NodeTypes.BUS]

        pyg_graph = Data(edge_index=edge_index, num_nodes=num_buses)
        nx_graph = to_networkx(pyg_graph)

        adjacency_list = get_adjacency_list(nx_graph)

        n_cuts, membership = pymetis.part_graph(self.num_clusters, adjacency=adjacency_list)

        virtual_x = torch.zeros((self.num_clusters, 1))

        virtual_to_bus = []
        bus_to_virtual = []
        for cluster_id in range(self.num_clusters):
            cluster_members = torch.argwhere(torch.tensor(membership) == cluster_id).T
            virtual_node = torch.ones_like(cluster_members) * cluster_id
            virtual_to_bus.append(torch.concatenate((virtual_node, cluster_members), dim=0))
            bus_to_virtual.append(torch.concatenate((cluster_members, virtual_node), dim=0))

        virtual_to_bus = torch.concatenate(virtual_to_bus, dim=1)
        bus_to_virtual = torch.concatenate(bus_to_virtual, dim=1)

        data[Virtual.NODE.value].x = virtual_x
        data[Virtual.NODE.value, Virtual.EDGE.value, NodeTypes.BUS.value].edge_index = virtual_to_bus
        data[NodeTypes.BUS.value, Virtual.EDGE.value, Virtual.NODE.value].edge_index = bus_to_virtual

        return data


@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="experiments")
def main(cfg: DictConfig):
    dataset = OPFDataset(
        "data", case_name=cfg.data.case_name, split="val", topological_perturbations=False, num_groups=1
    )

    data = dataset[0]
    edge_index = torch.concatenate(
        (
            data.edge_index_dict[NodeTypes.BUS, EdgeTypes.AC_LINE, NodeTypes.BUS],
            data.edge_index_dict[NodeTypes.BUS, EdgeTypes.TRANSFORMER, NodeTypes.BUS],
        ),
        dim=1,
    )
    edge_index = to_undirected(edge_index)

    num_buses = data.num_nodes_dict[NodeTypes.BUS]

    pyg_graph = Data(edge_index=edge_index, num_nodes=num_buses)
    nx_graph = to_networkx(pyg_graph)

    adjacency_list = get_adjacency_list(nx_graph)

    num_clusters = 4
    n_cuts, membership = pymetis.part_graph(num_clusters, adjacency=adjacency_list)

    available_colors_list = list(matplotlib.colors.TABLEAU_COLORS.keys())
    node_colors = [available_colors_list[cluster_id] for cluster_id in membership]

    pos = nx.nx_pydot.graphviz_layout(nx_graph, "neato")

    fig, axs = plt.subplots(1, 1)
    ax = axs
    nx.draw_networkx(nx_graph, pos, ax=ax, arrows=False, node_size=500, node_color=node_colors)
    ax.set_title("Power grid graph")
    ax.axis("off")  # remove border
    ax.set_aspect("auto")

    # transform = ClusterVirtualNodes(num_clusters=2)
    # transformed_data = transform(dataset[1])
    #
    # print("Transforming...")
    # transformed_dataset = OPFDataset(
    #     "data",
    #     case_name=cfg.data.case_name,
    #     split="val",
    #     topological_perturbations=False,
    #     num_groups=1,
    #     # pre_transform=None,
    #     # force_reload=True
    # )
    # print("Done transforming")
    # transformed_dataset_data = transformed_dataset[0]

    plt.show()


if __name__ == "__main__":
    main()
