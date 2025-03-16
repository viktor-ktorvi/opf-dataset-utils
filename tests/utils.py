from typing import Type
from unittest import TestCase

import torch
from hydra import compose, initialize
from torch_geometric.datasets import OPFDataset
from torch_geometric.loader import DataLoader


def setup_test(cls: Type[TestCase]):
    """
    Initialize test. Load config and test data.

    Returns
    -------
    """
    with initialize(version_base=None, config_path="../config"):
        cls.cfg = compose(config_name="tests")

    cls.device = torch.device(cls.cfg.device if torch.cuda.is_available() else "cpu")

    cls.loaders = []

    for case_name in cls.cfg.case_names:
        for topological_perturbations in [False, True]:
            dataset = OPFDataset(
                cls.cfg.data_directory,
                case_name=case_name,
                split="val",
                topological_perturbations=topological_perturbations,
            )
            cls.loaders.append(DataLoader(dataset, batch_size=cls.cfg.batch_size, shuffle=True))
