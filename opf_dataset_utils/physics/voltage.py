from typing import Dict

from torch import Tensor
from torch_geometric.data import HeteroData

from opf_dataset_utils.enumerations import (
    BusTypes,
    EdgeIndexIndices,
    GridBusIndices,
    NodeTypes,
    SolutionBusIndices,
)


def get_voltages_magnitudes(predictions: Dict) -> Tensor:
    """
    Get voltage magnitudes.
    Parameters
    ----------
    predictions: Dict
        Prediction dictionary.

    Returns
    -------
    voltage_magnitudes: Tensor
        Voltage magnitudes.
    """
    return predictions[NodeTypes.BUS][:, SolutionBusIndices.VOLTAGE_MAGNITUDE]


def get_voltage_angles(predictions: Dict) -> Tensor:
    """
    Get voltage angles.
    Parameters
    ----------
    predictions: Dict
        Prediction dictionary.

    Returns
    -------
    voltage_angles: Tensor
        Voltage angles.
    """
    return predictions[NodeTypes.BUS][:, SolutionBusIndices.VOLTAGE_ANGLE]


def get_reference_voltage_angles(data: HeteroData, predictions: Dict) -> Tensor:
    """
    Get the voltage angle(s) of the reference node(s).
    Parameters
    ----------
    data: HeteroData
        OPFData.
    predictions: Dict
        Prediction dictionary.

    Returns
    -------
    reference_voltage_angles: Tensor
        Reference voltage angles.
    """
    reference_mask = data.x_dict[NodeTypes.BUS][:, GridBusIndices.BUS_TYPE] == BusTypes.REFERENCE

    return get_voltage_angles(predictions)[reference_mask]


def calculate_voltage_angle_differences(data: HeteroData, predictions: Dict, branch_type: str) -> Tensor:
    """
    Calculate voltage angle differences.
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
    voltage_angle_differences: Tensor
        Voltage angle differences.
    """
    Va = get_voltage_angles(predictions)
    edge_index = data.edge_index_dict[(NodeTypes.BUS, branch_type, NodeTypes.BUS)]

    return Va[edge_index[EdgeIndexIndices.FROM]] - Va[edge_index[EdgeIndexIndices.TO]]
