from matplotlib import pyplot as plt
from torch_geometric.datasets import OPFDataset

from opf_dataset_utils.plotting.draw import draw_graph


def main():
    """
    Draw the graph of a sample from the OPFDataset.
    Returns
    -------

    """
    dataset = OPFDataset("data", case_name="pglib_opf_case14_ieee", split="val", num_groups=1)
    fig, ax = plt.subplots(1, 1)
    # TODO draw directed graphs as well
    draw_graph(dataset[0], ax=ax, node_size=300)

    plt.show()


if __name__ == "__main__":
    main()
