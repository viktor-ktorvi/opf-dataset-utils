from omegaconf import DictConfig
from torch_geometric.data.lightning import LightningDataset
from torch_geometric.datasets import OPFDataset
from torch_geometric.transforms import Compose

from opf_dataset_utils import DATA_DIR
from scripts.experiments.virtual_nodes import ClusterVirtualNodes


class OPFDataModule(LightningDataset):
    """Lightning data module for the OPFDataset."""

    batch_size: int
    dataset_train: OPFDataset
    dataset_val: OPFDataset
    dataset_test: OPFDataset

    def __init__(self, cfg: DictConfig):
        case_name = cfg.data.case_name
        topological_perturbations = cfg.data.topological_perturbations
        num_groups = cfg.data.num_groups
        force_reload = cfg.data.force_reload

        # TODO might wanna do ToUndirected here instead of dynamically in the model

        transform = Compose([ClusterVirtualNodes(num_clusters=cfg.training.num_virtual_nodes)])

        dataset_train = OPFDataset(
            DATA_DIR,
            case_name=case_name,
            num_groups=num_groups,
            topological_perturbations=topological_perturbations,
            split="train",
            pre_transform=transform,
            force_reload=force_reload,
        )

        dataset_val = OPFDataset(
            DATA_DIR,
            case_name=case_name,
            num_groups=num_groups,
            topological_perturbations=topological_perturbations,
            split="val",
            pre_transform=transform,
            force_reload=force_reload,
        )

        dataset_test = OPFDataset(
            DATA_DIR,
            case_name=case_name,
            num_groups=num_groups,
            topological_perturbations=topological_perturbations,
            split="test",
            pre_transform=transform,
            force_reload=force_reload,
        )

        super().__init__(
            train_dataset=dataset_train,
            val_dataset=dataset_val,
            test_dataset=dataset_test,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=False,
        )
