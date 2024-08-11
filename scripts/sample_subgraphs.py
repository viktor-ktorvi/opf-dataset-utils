import math
import os
import random

import hydra
import matplotlib
from matplotlib import pyplot as plt, gridspec
from omegaconf import DictConfig
from torch_geometric import seed_everything
from tqdm import tqdm

from opf_dataset_utils.data.loading import json2data
from opf_dataset_utils.data.subgraph_sampling import sample_buses, contains_reference_bus, is_within_size_range
from opf_dataset_utils.plotting.draw import draw_graph


@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(), "config"), config_name="sample_subgraphs")
def main(cfg: DictConfig):
    """
    Sample subgraphs according to config.
    Parameters
    ----------
    cfg: DictConfig


    Returns
    -------

    """
    seed_everything(cfg.random_seed)
    matplotlib.rcParams["figure.autolayout"] = True
    data = json2data(cfg.filepath)

    fig, ax = plt.subplots(1, 1)
    draw_graph(data, ax=ax, node_size=cfg.node_size)
    ax.set_title("Base graph")

    rows = int(math.ceil(cfg.num_subgraphs / cfg.num_columns))

    grid_spec = gridspec.GridSpec(rows, cfg.num_columns)
    fig = plt.figure(figsize=(cfg.inches_per.column * cfg.num_columns, cfg.inches_per.row * rows))

    for i in tqdm(range(cfg.num_subgraphs), desc="Sampling subgraphs"):
        while True:
            subset_dict = sample_buses(data,
                                       num_hops=random.randint(cfg.num_hops_range.min, cfg.num_hops_range.max),
                                       bus_retention_ratio=cfg.bus_retention_ratio,
                                       equipment_retention_ratio=cfg.equipment_retention_ratio)

            if contains_reference_bus(subset_dict, data) and is_within_size_range(subset_dict, minimum=cfg.bus_size_range.min, maximum=cfg.bus_size_range.max):
                break

        subgraph_data = data.subgraph(subset_dict)

        ax = fig.add_subplot(grid_spec[i])
        draw_graph(subgraph_data, ax=ax, node_size=cfg.node_size, show_legend=False)

    fig.savefig(cfg.saving.filepath)

    if cfg.show:
        plt.show()


if __name__ == "__main__":
    main()
