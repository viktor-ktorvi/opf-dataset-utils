from torch_geometric.data.lightning import LightningDataset
from torch_geometric.datasets import OPFDataset

from opf_dataset_utils import DATA_DIR


class OPFDataModule(LightningDataset):
    """
    Lightning data module for the OPFDataset.
    """

    batch_size: int
    dataset_train: OPFDataset
    dataset_val: OPFDataset
    dataset_test: OPFDataset

    def __init__(
        self,
        case_name: str,
        topological_perturbations: bool = True,
        num_groups: int = 1,
        batch_size: int = 32,
        num_workers: int = 1,
    ):
        # TODO Do we wanna transform the data - e.g., some edges are (I think) directed, and maybe we wanna change that.
        dataset_train = OPFDataset(
            DATA_DIR,
            case_name=case_name,
            num_groups=num_groups,
            topological_perturbations=topological_perturbations,
            split="train",
        )

        dataset_val = OPFDataset(
            DATA_DIR,
            case_name=case_name,
            num_groups=num_groups,
            topological_perturbations=topological_perturbations,
            split="val",
        )

        dataset_test = OPFDataset(
            DATA_DIR,
            case_name=case_name,
            num_groups=num_groups,
            topological_perturbations=topological_perturbations,
            split="test",
        )

        super().__init__(
            train_dataset=dataset_train,
            val_dataset=dataset_val,
            test_dataset=dataset_test,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=False,
        )
