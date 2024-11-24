import torch
from torch import Tensor, nn


class StandardScaler(nn.Module):
    """
    A tensor standard scaler.

    Calculates the mean and standard deviation of tensors by being iteratively fed batches of tensors, and calculating
    the statistics once the entire (train) dataset had been fed through.
    """

    mean: Tensor
    std: Tensor

    def __init__(self, inverse: bool = False):
        super().__init__()

        self.value_sum = None
        self.square_sum = None
        self.num_samples = 0

        self.inverse = inverse

    def update(self, batch: Tensor):
        size = batch.shape[-1]
        if self.value_sum is None:
            self.value_sum = torch.zeros((size,))
            self.square_sum = torch.zeros((size,))

        batch = batch.reshape(-1, size)
        self.num_samples += batch.shape[0]
        self.value_sum += batch.sum(dim=0)
        self.square_sum += (batch**2).sum(dim=0)

    def calculate_statistics(self):
        mean = self.value_sum / self.num_samples

        std = torch.sqrt(self.square_sum / self.num_samples - mean**2)
        std[std < 1e-9] = 1.0
        std[torch.isnan(std)] = 1.0

        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def scale(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.std

    def inverse_scale(self, x: Tensor) -> Tensor:
        return x * self.std + self.mean

    def forward(self, x: Tensor) -> Tensor:
        if self.inverse:
            return self.inverse_scale(x)

        return self.scale(x)


class HeteroStandardScaler(nn.Module):
    """A tensor standard scaler for hetero data."""

    def __init__(self, inverse: bool = False):
        super().__init__()
        self.inverse = inverse
        self.scalers = None

    def update(self, batch_dict: dict[str, Tensor]):
        if self.scalers is None:
            scalers = {}
            for node_type in batch_dict:
                scalers[node_type] = StandardScaler(inverse=self.inverse)

            self.scalers = nn.ModuleDict(scalers)
        for node_type in self.scalers:
            self.scalers[node_type].update(batch_dict[node_type])

    def calculate_statistics(self):
        for node_type in self.scalers:
            self.scalers[node_type].calculate_statistics()

    def scale(self, x_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        out_dict = {}
        for node_type in self.scalers:
            out_dict[node_type] = self.scalers[node_type].scale(x_dict[node_type])

        return out_dict

    def inverse_scale(self, x_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        out_dict = {}
        for node_type in self.scalers:
            out_dict[node_type] = self.scalers[node_type].inverse_scale(x_dict[node_type])

        return out_dict

    def forward(self, x_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        out_dict = {}
        for node_type in self.scalers:
            out_dict[node_type] = self.scalers[node_type](x_dict[node_type])

        return out_dict
