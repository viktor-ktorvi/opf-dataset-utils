from typing import Dict

import torch
from torch import Tensor
from torch_geometric.data import Batch, HeteroData

from opf_dataset_utils.enumerations import (
    GridGeneratorIndices,
    NodeTypes,
    SolutionGeneratorIndices,
)


def calculate_costs_per_generator(data: HeteroData, predictions: Dict) -> Tensor:
    """
    Calculate costs per generator.

    Parameters
    ----------
    data: HeteroData
        OPFData.
    predictions: Dict
        Predictions dictionary.

    Returns
    -------
    costs_per_generator: Tensor
        Costs per generator [$/h].
    """
    Pg = predictions[NodeTypes.GENERATOR][:, SolutionGeneratorIndices.ACTIVE_POWER]
    c2 = data.x_dict[NodeTypes.GENERATOR][:, GridGeneratorIndices.COST_SQUARED]
    c1 = data.x_dict[NodeTypes.GENERATOR][:, GridGeneratorIndices.COST_LINEAR]
    c0 = data.x_dict[NodeTypes.GENERATOR][:, GridGeneratorIndices.COST_OFFSET]

    return c2 * Pg**2 + c1 * Pg + c0


def calculate_costs_per_grid(data: HeteroData, predictions: Dict) -> Tensor:
    """
    Calculate costs per grid.

    Parameters
    ----------
    data: HeteroData
        OPFData.
    predictions: Dict
        Predictions dictionary.

    Returns
    -------
    costs_per_grid: Tensor
        Costs per grid [$/h].
    """
    costs_per_generator = calculate_costs_per_generator(data, predictions)

    if not isinstance(data, Batch):
        return costs_per_generator.sum()

    return torch.zeros(
        torch.unique(data.batch_dict[NodeTypes.GENERATOR]).shape, device=costs_per_generator.device
    ).scatter_add_(dim=0, index=data.batch_dict[NodeTypes.GENERATOR], src=costs_per_generator)
