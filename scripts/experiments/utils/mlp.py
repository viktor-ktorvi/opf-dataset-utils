from typing import Any

from torch import Tensor, nn
from torch_geometric.nn import MLP


class HeteroMLP(nn.Module):
    """
    An MLP that takes in heterogeneous data in the for of a dictionary.
    """

    def __init__(
        self, in_channels: dict[Any, int], out_channels: dict[Any, int], hidden_channels: int, num_layers: int
    ):
        super().__init__()
        mlps = {}
        for key in in_channels:
            mlps[key] = MLP(
                in_channels=in_channels[key],
                out_channels=out_channels[key],
                hidden_channels=hidden_channels,
                num_layers=num_layers,
            )
        self.mlp = nn.ModuleDict(mlps)

    def forward(self, x_dict: dict[Any, Tensor]) -> dict[Any, Tensor]:
        out = {}
        for key in self.mlp:
            out[key] = self.mlp[key](x_dict[key])

        return out
