import warnings
from enum import StrEnum

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch import LongTensor, Tensor, nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import HeteroData
from torch_geometric.nn import GAT, to_hetero
from torchmetrics import Metric, MetricCollection, R2Score

import wandb
from opf_dataset_utils import CONFIG_PATH
from opf_dataset_utils.physics.metrics.aggregation import AggregationTypes
from opf_dataset_utils.physics.metrics.cost import OptimalityGap
from opf_dataset_utils.physics.metrics.inequality.bound_types import BoundTypes
from opf_dataset_utils.physics.metrics.inequality.voltage import (
    VoltageAngleDifferenceInequalityError,
    VoltageMagnitudeInequalityError,
)
from opf_dataset_utils.physics.metrics.power import Power, PowerTypes
from opf_dataset_utils.physics.metrics.power_flow import PowerFlowError
from scripts.experiments.utils.data import OPFDataModule
from scripts.experiments.utils.mlp import HeteroMLP
from scripts.experiments.utils.standard_scaler import HeteroStandardScaler


class Split(StrEnum):
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"


def create_opf_metrics(split: str) -> MetricCollection:
    metric_dict = {}
    for aggr in AggregationTypes:
        metric_dict[f"{split}/{aggr} optimality gap [%]"] = OptimalityGap(aggr=aggr)

        for power_type in PowerTypes:
            metric_dict[f"{split}/{aggr} absolute {power_type} power flow error [kVA]"] = PowerFlowError(
                aggr=aggr, power_type=power_type, unit="kilo", value_type="absolute"
            )
            metric_dict[f"{split}/{aggr} relative {power_type} power flow error [%]"] = PowerFlowError(
                aggr=aggr, power_type=power_type, value_type="relative"
            )
            metric_dict[f"{split}/{aggr} {power_type} power [kVA]"] = Power(
                aggr=aggr, power_type=power_type, unit="kilo"
            )

        for bound_type in BoundTypes:
            metric_dict[
                f"{split}/{aggr} absolute {bound_type} voltage magnitude inequality error [per-unit]"
            ] = VoltageMagnitudeInequalityError(aggr=aggr, bound_type=bound_type, value_type="absolute")
            metric_dict[
                f"{split}/{aggr} relative {bound_type} voltage magnitude inequality error [%]"
            ] = VoltageMagnitudeInequalityError(aggr=aggr, bound_type=bound_type, value_type="relative")

            metric_dict[
                f"{split}/{aggr} absolute {bound_type} voltage angle difference error [deg]"
            ] = VoltageAngleDifferenceInequalityError(
                aggr=aggr, bound_type=bound_type, value_type="absolute", unit="degree"
            )
            metric_dict[
                f"{split}/{aggr} relative {bound_type} voltage angle difference error [%]"
            ] = VoltageAngleDifferenceInequalityError(aggr=aggr, bound_type=bound_type, value_type="relative")

    return MetricCollection(metric_dict)


class ExampleModel(LightningModule):
    """
    An example (lightning) model for the OPFDataset.

    Processes the heterogeneous data by normalizing the inputs for each node type,
    projecting the features to a hidden dimension using an MLP,
    processing the data with a GNN, projects the outputs down to the target dimensions,
    and inverse normalizes the outputs.

    Calculates and logs metrics like R2 score and MSE, as well as OPF specific metrics
    like absolute power flow error.

    Includes the absolute power flow error as a penalty in the loss.
    """

    learning_rate: float

    in_scaler: HeteroStandardScaler
    in_mlp: HeteroMLP
    gnn: nn.Module
    out_mlp: HeteroMLP
    out_scaler: HeteroStandardScaler

    criterion: nn.Module
    mean_abs_power_flow_errors: dict[str, Metric]
    r2_scores: dict[str, dict[str, Metric]]

    def __init__(
        self,
        data_module: OPFDataModule,
        hidden_channels: int,
        num_layers: int,
        num_mlp_layers: int,
        learning_rate: float,
        power_flow_multiplier: float,
        heads: int,
    ):
        super().__init__()

        self.learning_rate = learning_rate
        self.power_flow_multiplier = power_flow_multiplier
        self.criterion = nn.MSELoss()  # TODO probabilistic forecasting

        # init modules
        example_batch: HeteroData = next(iter(data_module.train_dataloader()))

        self.in_scaler = HeteroStandardScaler()
        self.out_scaler = HeteroStandardScaler(inverse=True)
        for batch in data_module.train_dataloader():
            self.in_scaler.update(batch.x_dict)
            self.out_scaler.update(batch.y_dict)
        self.in_scaler.calculate_statistics()
        self.out_scaler.calculate_statistics()

        self.in_mlp = HeteroMLP(
            in_channels=example_batch.num_node_features,
            out_channels={key: hidden_channels for key in example_batch.x_dict},
            hidden_channels=hidden_channels,
            num_layers=num_mlp_layers,
        )

        self.gnn = to_hetero(
            GAT(
                in_channels=hidden_channels,
                edge_dim=-1,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                out_channels=hidden_channels,
                add_self_loops=False,
                jk="cat",
                v2=True,
                heads=heads,
            ),
            example_batch.metadata(),
        )

        self.out_mlp = HeteroMLP(
            in_channels={key: hidden_channels for key in example_batch.y_dict},
            out_channels={key: y.shape[-1] for key, y in example_batch.y_dict.items()},
            hidden_channels=hidden_channels,
            num_layers=num_mlp_layers,
        )

        # initialize lazy modules (edge_dim in the GNN)
        with torch.no_grad():
            self(example_batch.x_dict, example_batch.edge_index_dict, example_batch.edge_attr_dict)

        # TODO metrics for:
        #  inequality constraints -- absolute and relative (divided by the range?)
        #  thresholded relative mean-absolute error (TRMAE)? (maybe threshold or maybe just ignore zeros) (for Pg, Qg, Sg?, vm, va, Sf?, St?)

        # metrics (each has to be an attribute of the model class)
        for split in Split:
            # OPF related metrics
            setattr(self, f"{split}_opf_metrics", create_opf_metrics(split))

            # R2 score
            for node_type in example_batch.y_dict:
                setattr(self, f"{split}_{node_type}_r2", R2Score(num_outputs=example_batch.y_dict[node_type].shape[-1]))

        self.opf_metrics = {split: getattr(self, f"{split}_opf_metrics") for split in Split}

        self.r2_scores = {
            split: {node_type: getattr(self, f"{split}_{node_type}_r2") for node_type in example_batch.y_dict}
            for split in Split
        }

    def forward(
        self,
        x_dict: dict[str, Tensor],
        edge_index_dict: dict[tuple[str, str, str], LongTensor],
        edge_attr_dict: dict[tuple[str, str, str], Tensor],
    ) -> dict[str, Tensor]:
        h_dict = self.in_scaler(x_dict)
        h_dict = self.in_mlp(h_dict)

        h_dict = self.gnn(x=h_dict, edge_index=edge_index_dict, edge_attr=edge_attr_dict)

        h_dict = self.out_mlp(h_dict)
        out_dict = self.out_scaler(h_dict)

        return out_dict

    def _calculate_loss(self, pred_dict: dict[str, Tensor], y_dict: dict[str, Tensor]) -> Tensor:
        pred_dict_scaled = self.out_scaler.scale(pred_dict)
        y_dict_scaled = self.out_scaler.scale(y_dict)
        return torch.stack([self.criterion(pred_dict_scaled[key], y_dict_scaled[key]) for key in y_dict]).sum()

    def _shared_eval(self, batch, split: str):
        logging_kwargs = dict(on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=batch.batch_size)

        # predict
        pred_dict = self(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)

        # calculate loss
        mse = self._calculate_loss(pred_dict, batch.y_dict)
        opf_metrics: dict[str, Tensor] = self.opf_metrics[split](batch, pred_dict)
        apparent_power_flow_error = opf_metrics[
            f"{split}/{AggregationTypes.MEAN} absolute {PowerTypes.APPARENT} power flow error [kVA]"
        ]

        loss = mse + self.power_flow_multiplier * apparent_power_flow_error

        # log metrics
        self.log(f"{split}/MSE", mse, **logging_kwargs)
        self.log(f"{split}/loss", loss, **logging_kwargs)
        self.log_dict(opf_metrics, **logging_kwargs)

        for node_type in batch.y_dict:
            # r2 score
            self.r2_scores[split][node_type](pred_dict[node_type], batch.y_dict[node_type])
            self.log(f"{split}/{node_type}_r2", self.r2_scores[split][node_type], **logging_kwargs)

        return loss

    def training_step(self, batch, batch_idx) -> Tensor:
        return self._shared_eval(batch, Split.TRAIN)

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, Split.VALIDATION)

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, Split.TEST)

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer=optimizer, factor=0.5, patience=20, threshold=0.01, threshold_mode="rel", min_lr=1e-5
                ),
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val/MSE",
                "strict": True,
                "name": None,
            },
        }


@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="experiments")
def main(cfg: DictConfig):
    """
    Train a GNN on the OPFDataset.
    Parameters
    ----------
    cfg: DictConfig
        Config.

    Returns
    -------

    """
    warnings.filterwarnings("ignore", message="The total number of parameters detected may be inaccurate")
    warnings.filterwarnings("ignore", message="There is a wandb run already in progress")

    wandb.init(project=cfg.wandb.project, mode=cfg.wandb.mode)

    # cross-update hydra and wandb configs
    cfg = OmegaConf.merge(cfg, dict(wandb.config))
    wandb.config.update(
        {"config": OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)}, allow_val_change=True
    )

    wandb_logger = WandbLogger(project=cfg.wandb.project, offline=False if cfg.wandb.mode == "online" else True)

    torch.set_float32_matmul_precision("high")

    torch.multiprocessing.set_sharing_strategy("file_system")
    seed_everything(cfg.random_seed, workers=True)

    opf_data = OPFDataModule(
        case_name=cfg.data.case_name,
        topological_perturbations=cfg.data.topological_perturbations,
        num_groups=cfg.data.num_groups,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.num_workers,
    )

    model = ExampleModel(
        opf_data,
        hidden_channels=cfg.training.hidden_channels,
        num_layers=cfg.training.num_layers,
        num_mlp_layers=cfg.training.num_mlp_layers,
        learning_rate=cfg.training.learning_rate,
        power_flow_multiplier=cfg.training.power_flow_multiplier,
        heads=cfg.training.heads,
    )

    learning_rate_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = Trainer(
        deterministic=True,
        accelerator=cfg.training.accelerator,
        max_epochs=cfg.training.epochs,
        gradient_clip_val=cfg.training.gradient_clip_val,
        logger=wandb_logger,
        callbacks=[learning_rate_monitor],
    )

    trainer.fit(model, opf_data)


if __name__ == "__main__":
    main()
