from typing import Dict

from torch import Tensor
from torch_geometric.data import HeteroData

from opf_dataset_utils.enumerations import GridBusIndices, NodeTypes
from opf_dataset_utils.physics.errors.inequality.violations import (
    calculate_lower_violations,
    calculate_upper_violations,
)
from opf_dataset_utils.physics.utils import get_branch_type_indices
from opf_dataset_utils.physics.voltage import (
    calculate_voltage_angle_differences,
    get_voltages_magnitudes,
)


def calculate_upper_voltage_magnitude_errors(data: HeteroData, predictions: Dict) -> Tensor:
    """
    Calculate upper voltage magnitude errors.
    Parameters
    ----------
    data: HeteroData
        OPFData.
    predictions: Dict
        Prediction dictionary.

    Returns
    -------
    upper_voltage_magnitude_errors: Tensor
        Upper voltage magnitude errors.
    """
    Vm = get_voltages_magnitudes(predictions)
    Vm_max = data.x_dict[NodeTypes.BUS][:, GridBusIndices.VOLTAGE_MAX]

    return calculate_upper_violations(Vm, Vm_max)


def calculate_lower_voltage_magnitude_errors(data: HeteroData, predictions: Dict) -> Tensor:
    """
    Calculate lower voltage magnitude errors.
    Parameters
    ----------
    data: HeteroData
        OPFData.
    predictions: Dict
        Prediction dictionary.

    Returns
    -------
    lower_voltage_magnitude_errors: Tensor
        Lower voltage magnitude errors.
    """
    Vm = get_voltages_magnitudes(predictions)
    Vm_min = data.x_dict[NodeTypes.BUS][:, GridBusIndices.VOLTAGE_MIN]

    return calculate_lower_violations(Vm, Vm_min)


def calculate_upper_voltage_angle_difference_errors(data: HeteroData, predictions: Dict, branch_type: str) -> Tensor:
    """
    Calculate upper voltage error difference errors for the given branch type.
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
    upper_voltage_angle_difference_errors: Tensor
        Upper voltage angle difference errors.
    """
    Va_diffs = calculate_voltage_angle_differences(data, predictions, branch_type)

    indices = get_branch_type_indices(branch_type)
    Va_diff_max = data.edge_attr_dict[NodeTypes.BUS, branch_type, NodeTypes.BUS][:, indices.ANGLE_DIFF_MAX]

    return calculate_upper_violations(Va_diffs, Va_diff_max)


def calculate_lower_voltage_angle_difference_errors(data: HeteroData, predictions: Dict, branch_type: str) -> Tensor:
    """
    Calculate lower voltage error difference errors for the given branch type.
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
    lower_voltage_angle_difference_errors: Tensor
        Lower voltage angle difference errors.
    """
    Va_diffs = calculate_voltage_angle_differences(data, predictions, branch_type)

    indices = get_branch_type_indices(branch_type)
    Va_diff_min = data.edge_attr_dict[NodeTypes.BUS, branch_type, NodeTypes.BUS][:, indices.ANGLE_DIFF_MIN]

    return calculate_lower_violations(Va_diffs, Va_diff_min)
