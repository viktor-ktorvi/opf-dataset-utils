from typing import Dict

from torch import Tensor
from torch_geometric.data import HeteroData

from opf_dataset_utils.enumerations import NodeTypes
from opf_dataset_utils.physics.errors.inequality.violations import (
    calculate_upper_violations,
)
from opf_dataset_utils.physics.power import calculate_branch_powers
from opf_dataset_utils.physics.utils import get_branch_type_indices


def calculate_branch_power_errors_from(data: HeteroData, predictions: Dict, branch_type: str) -> Tensor:
    """
    Calculate upper branch power errors for the given branch type.
    Parameters
    ----------
    data: HeteroData
        OPFData.
    predictions: Dict
        Prediction dictionary.
    branch_type: str
        One of ['ac_line', 'transformer'].

    Returns
    -------
    upper_branch_power_difference_errors: Tensor
        Upper branch power difference errors.
    """
    S_ij, _ = calculate_branch_powers(data, predictions, branch_type)

    indices = get_branch_type_indices(branch_type)
    S_max = data.edge_attr_dict[NodeTypes.BUS, branch_type, NodeTypes.BUS][:, indices.LONG_TERM_RATING]

    return calculate_upper_violations(S_ij.abs(), S_max)


def calculate_branch_power_errors_to(data: HeteroData, predictions: Dict, branch_type: str) -> Tensor:
    """
    Calculate lower branch power errors for the given branch type.
    Parameters
    ----------
    data: HeteroData
        OPFData.
    predictions: Dict
        Prediction dictionary.
    branch_type: str
        One of ['ac_line', 'transformer'].

    Returns
    -------
    lower_branch_power_difference_errors: Tensor
        Lower branch power difference errors.
    """
    _, S_ji = calculate_branch_powers(data, predictions, branch_type)

    indices = get_branch_type_indices(branch_type)
    S_max = data.edge_attr_dict[NodeTypes.BUS, branch_type, NodeTypes.BUS][:, indices.LONG_TERM_RATING]

    return calculate_upper_violations(S_ji.abs(), S_max)
