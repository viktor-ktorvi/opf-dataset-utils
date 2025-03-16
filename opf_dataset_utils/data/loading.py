import json
from typing import Dict

import torch
from torch import LongTensor
from torch_geometric.data import HeteroData


def extract_edge_index(data_json: Dict, edge_name: str) -> LongTensor:
    """
    Extract the edge index from the data JSON.

    Parameters
    ----------
    data_json: Dict
        Data JSON.
    edge_name: str
        Edge name.

    Returns
    -------
    edge_index: LongTensor
        Edge index.
    """
    return torch.LongTensor(
        [
            data_json["grid"]["edges"][edge_name]["senders"],
            data_json["grid"]["edges"][edge_name]["receivers"],
        ]
    )


def extract_edge_index_rev(data_json: Dict, edge_name: str) -> LongTensor:
    """
    Extract the reverse edge index from the data JSON.

    Parameters
    ----------
    data_json: Dict
        Data JSON.
    edge_name: str
        Edge name.

    Returns
    -------
    edge_index: LongTensor
        Edge index.
    """
    return torch.LongTensor(
        [
            data_json["grid"]["edges"][edge_name]["receivers"],
            data_json["grid"]["edges"][edge_name]["senders"],
        ]
    )


def json2data(filepath: str) -> HeteroData:
    """
    Load a JSON object and convert it to OPFData.

    Parameters
    ----------
    filepath: str
        File path.

    Returns
    -------
    data: HeteroData
        OPFData.
    """
    with open(filepath) as f:
        obj = json.load(f)

    grid = obj["grid"]
    solution = obj["solution"]

    # Graph-level properties:
    data = HeteroData()
    data.x = torch.tensor(grid["context"]).view(-1)

    # Nodes (only some have a target):
    data["bus"].x = torch.tensor(grid["nodes"]["bus"])
    data["bus"].y = torch.tensor(solution["nodes"]["bus"])

    data["generator"].x = torch.tensor(grid["nodes"]["generator"])
    data["generator"].y = torch.tensor(solution["nodes"]["generator"])

    data["load"].x = torch.tensor(grid["nodes"]["load"])

    data["shunt"].x = torch.tensor(grid["nodes"]["shunt"])

    # Edges (only ac lines and transformers have features):
    data["bus", "ac_line", "bus"].edge_index = extract_edge_index(obj, "ac_line")  #
    data["bus", "ac_line", "bus"].edge_attr = torch.tensor(grid["edges"]["ac_line"]["features"])
    data["bus", "ac_line", "bus"].edge_label = torch.tensor(solution["edges"]["ac_line"]["features"])

    data["bus", "transformer", "bus"].edge_index = extract_edge_index(obj, "transformer")  #
    data["bus", "transformer", "bus"].edge_attr = torch.tensor(grid["edges"]["transformer"]["features"])
    data["bus", "transformer", "bus"].edge_label = torch.tensor(solution["edges"]["transformer"]["features"])

    data["generator", "generator_link", "bus"].edge_index = extract_edge_index(obj, "generator_link")  #
    data["bus", "generator_link", "generator"].edge_index = extract_edge_index_rev(obj, "generator_link")  #

    data["load", "load_link", "bus"].edge_index = extract_edge_index(obj, "load_link")  #
    data["bus", "load_link", "load"].edge_index = extract_edge_index_rev(obj, "load_link")  #

    data["shunt", "shunt_link", "bus"].edge_index = extract_edge_index(obj, "shunt_link")  #
    data["bus", "shunt_link", "shunt"].edge_index = extract_edge_index_rev(obj, "shunt_link")  #

    return data
