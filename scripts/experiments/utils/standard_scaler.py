import torch
from torch import Tensor, nn
from torch_geometric.loader import DataLoader


class HeteroStandardScaler(nn.Module):
    """
    A standard scaler that takes in heterogeneous data.
    """

    def __init__(self, train_loader: DataLoader, attribute: str, inverse: bool = False):
        super().__init__()

        self.inverse = inverse

        class Aggregator:
            def __init__(self, size: int):
                self.value_sum = torch.zeros((size,))
                self.square_sum = torch.zeros((size,))
                self.num_samples = 0

            def update(self, batch: Tensor):
                self.num_samples += batch.shape[0]
                self.value_sum += batch.sum(dim=0)
                self.square_sum += (batch**2).sum(dim=0)

        aggregators: dict[str, Aggregator] = {}

        # calculate mean and std by summing up values and their squares
        for train_batch in train_loader:
            values_dict = getattr(train_batch, f"{attribute}_dict")

            for node_type in values_dict:
                values = values_dict[node_type]
                if node_type not in aggregators:
                    aggregators[node_type] = Aggregator(size=values.shape[-1])

                aggregators[node_type].update(values)

        for node_type in aggregators:
            aggr = aggregators[node_type]
            mean = aggr.value_sum / aggr.num_samples
            std = torch.sqrt(aggr.square_sum / aggr.num_samples - mean**2)

            std[std < 1e-9] = 1.0
            std[torch.isnan(std)] = 1.0

            self.register_buffer(self.mean_name(node_type), mean)
            self.register_buffer(self.std_name(node_type), std)

    @staticmethod
    def mean_name(node_type: str) -> str:
        return f"mean_{node_type}"

    @staticmethod
    def std_name(node_type: str) -> str:
        return f"std_{node_type}"

    def get_mean(self, node_type: str) -> Tensor:
        return getattr(self, self.mean_name(node_type))

    def get_std(self, node_type: str) -> Tensor:
        return getattr(self, self.std_name(node_type))

    def scale(self, x_dict: dict[str, Tensor], node_type: str) -> Tensor:
        x = x_dict[node_type]
        mean = self.get_mean(node_type)
        std = self.get_std(node_type)
        if self.inverse:
            return std * x + mean

        return (x - mean) / std

    def forward(self, x_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        out = {}
        for node_type in x_dict:
            out[node_type] = self.scale(x_dict, node_type)

        return out
