from typing import Dict

from torch import Tensor
from torch_geometric.data import HeteroData

from opf_dataset_utils.enumerations import (
    GridGeneratorIndices,
    NodeTypes,
    SolutionGeneratorIndices,
)
from opf_dataset_utils.errors.inequality.violations import (
    calculate_lower_violations,
    calculate_upper_violations,
)


def calculate_upper_active_power_errors(data: HeteroData, predictions: Dict) -> Tensor:
    """
    Calculate upper active power errors.

    Parameters
    ----------
    data: HeteroData
        OPFData.
    predictions: Dict
        Prediction dictionary.

    Returns
    -------
    upper_active_power_errors: Tensor
        Upper active power errors.
    """
    Pg = predictions[NodeTypes.GENERATOR][:, SolutionGeneratorIndices.ACTIVE_POWER]
    Pg_max = data.x_dict[NodeTypes.GENERATOR][:, GridGeneratorIndices.ACTIVE_POWER_MAX]

    return calculate_upper_violations(Pg, Pg_max)


def calculate_lower_active_power_errors(data: HeteroData, predictions: Dict) -> Tensor:
    """
    Calculate lower active power errors.

    Parameters
    ----------
    data: HeteroData
        OPFData.
    predictions: Dict
        Prediction dictionary.

    Returns
    -------
    lower_active_power_errors: Tensor
        Lower active power errors.
    """
    Pg = predictions[NodeTypes.GENERATOR][:, SolutionGeneratorIndices.ACTIVE_POWER]
    Pg_min = data.x_dict[NodeTypes.GENERATOR][:, GridGeneratorIndices.ACTIVE_POWER_MIN]

    return calculate_lower_violations(Pg, Pg_min)


def calculate_upper_reactive_power_errors(data: HeteroData, predictions: Dict) -> Tensor:
    """
    Calculate upper reactive power errors.

    Parameters
    ----------
    data: HeteroData
        OPFData.
    predictions: Dict
        Prediction dictionary.

    Returns
    -------
    upper_reactive_power_errors: Tensor
        Upper reactive power errors.
    """
    Qg = predictions[NodeTypes.GENERATOR][:, SolutionGeneratorIndices.REACTIVE_POWER]
    Qg_max = data.x_dict[NodeTypes.GENERATOR][:, GridGeneratorIndices.REACTIVE_POWER_MAX]

    return calculate_upper_violations(Qg, Qg_max)


def calculate_lower_reactive_power_errors(data: HeteroData, predictions: Dict) -> Tensor:
    """
    Calculate lower reactive power errors.

    Parameters
    ----------
    data: HeteroData
        OPFData.
    predictions: Dict
        Prediction dictionary.

    Returns
    -------
    lower_reactive_power_errors: Tensor
        Lower reactive power errors.
    """
    Qg = predictions[NodeTypes.GENERATOR][:, SolutionGeneratorIndices.REACTIVE_POWER]
    Qg_min = data.x_dict[NodeTypes.GENERATOR][:, GridGeneratorIndices.REACTIVE_POWER_MIN]

    return calculate_lower_violations(Qg, Qg_min)
