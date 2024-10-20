import glob
import os
import random

import hydra
import matplotlib
import networkx as nx
import numpy as np
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from torch_geometric import seed_everything
from torch_geometric.data import HeteroData
from torch_geometric.datasets import OPFDataset
from torch_geometric.utils import to_networkx
from tqdm import tqdm

from opf_dataset_utils.data.loading import json2data
from opf_dataset_utils.data.subgraph_sampling import sample_buses, contains_reference_bus, is_within_size_range


def to_homogeneous_graph(data: HeteroData) -> nx.Graph:
    """
    Convert a hetero graph to a homogeneous networkx graph.
    Parameters
    ----------
    data: HeteroData
        Hetero graph.

    Returns
    -------
    homogeneous_graph: nx.Graph
        Homogeneous graph.
    """
    return to_networkx(data.to_homogeneous())


def sample_subgraphs(num_samples: int, cfg: DictConfig) -> tuple[list[nx.Graph], list[HeteroData]]:
    """
    Sample subgraphs from a larger graph using k-hop neighborhoods and random retention.
    Parameters
    ----------
    num_samples: int
        Number subgraphs to sample.
    cfg: DictConfig
        Config.

    Returns
    -------
    subgraphs: list[nx.Graph]
        List of subgraphs as networkx graphs.
    data_list: list[HeteroData]
        List of subgraphs as hetero data.
    """
    data = json2data(cfg.filepath)
    subgraphs = []
    data_list = []
    for _ in tqdm(range(num_samples), desc="Sampling subgraphs"):
        while True:
            subset_dict = sample_buses(data,
                                       num_hops=random.randint(cfg.num_hops_range.min, cfg.num_hops_range.max),
                                       bus_retention_ratio=cfg.bus_retention_ratio,
                                       equipment_retention_ratio=cfg.equipment_retention_ratio)

            if contains_reference_bus(subset_dict, data) and is_within_size_range(subset_dict, minimum=cfg.bus_size_range.min, maximum=cfg.bus_size_range.max):
                break

        subgraph_data = data.subgraph(subset_dict)
        data_list.append(subgraph_data)
        subgraphs.append(to_homogeneous_graph(subgraph_data))

    return subgraphs, data_list


def sample_OPFDataset_topologies(num_samples: int) -> tuple[list[nx.Graph], list[HeteroData]]:
    """
    Get a sample of the different topologies from the OPFDataset.
    Parameters
    ----------
    num_samples: int
        Number of samples.

    Returns
    -------
    graphs: list[nx.Graph]
        List of samples as networkx graphs.
    data_list: list[HeteroData]
        List of data as hetero data.
    """
    _ = OPFDataset(
        "data", case_name="pglib_opf_case30_ieee", split="val", topological_perturbations=True, num_groups=1
    )
    directory = "data/dataset_release_1_nminusone/pglib_opf_case30_ieee/raw/gridopt-dataset-tmp/dataset_release_1_nminusone/pglib_opf_case30_ieee"
    subdirectories = glob.glob(os.path.join(directory, "*/"))

    filepaths = []
    for subdir in subdirectories:
        filepaths += glob.glob(os.path.join(subdir, "*.json"))

    random.shuffle(filepaths)

    graphs = []
    data_list = []
    for filepath in filepaths[:num_samples]:
        data = json2data(filepath)
        data_list.append(data)
        graphs.append(to_homogeneous_graph(data))

    return graphs, data_list


@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(), "config"), config_name="sample_subgraphs")
def main(cfg: DictConfig):
    """
    Approximate the (within-dataset) edit distances for:
    * either subgraphs samples from a larger grid
    * or for topologies from OPFDataset.
    Parameters
    ----------
    cfg: DictConfig
        Config.

    Returns
    -------

    """
    seed_everything(cfg.random_seed)
    matplotlib.rcParams["figure.autolayout"] = True
    num_samples = cfg.edit_distance.num_samples

    if cfg.edit_distance.subgraphs:
        graphs, data_list = sample_subgraphs(num_samples, cfg)
    else:
        graphs, data_list = sample_OPFDataset_topologies(num_samples)  # TODO kinda wanna visualize the ones that have a large edit distance

    index_pairs = []
    for i in range(num_samples - 1):
        for j in range(i + 1, num_samples):
            index_pairs.append((i, j))

    def approximate_edit_distance(indices: tuple[int, int]) -> int:
        approx = None
        for step in range(cfg.edit_distance.approximation_iterations):
            approx = next(iter(nx.optimize_graph_edit_distance(graphs[indices[0]], graphs[indices[1]])))
        return approx

    edit_distance_approximations = Parallel(n_jobs=8)(delayed(approximate_edit_distance)(ij) for ij in tqdm(index_pairs, desc="Approximating edit distances"))

    fig, ax = plt.subplots(1, 1)
    ax.hist(edit_distance_approximations, bins=cfg.edit_distance.num_bins)
    ax.set_xlabel(r"approx. $GED(G_i, G_j)$")
    ax.set_ylabel("# graph pairs")

    print(f"Mean: {np.mean(edit_distance_approximations):5.1f}")
    print(f"Std: {np.std(edit_distance_approximations):5.1f}")

    if cfg.show:
        plt.show()


if __name__ == "__main__":
    main()
