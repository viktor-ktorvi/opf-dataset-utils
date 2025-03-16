from typing import List
from unittest import TestCase

import torch
from omegaconf import DictConfig
from torch_geometric.loader import DataLoader

from opf_dataset_utils.costs import calculate_costs_per_grid
from tests.utils import setup_test


class TestCosts(TestCase):
    cfg: DictConfig
    device: torch.device
    loaders: List[DataLoader]

    @classmethod
    def setUpClass(cls):
        """
        Initialize test. Load config and test data.

        Returns
        -------
        """
        setup_test(cls)

    def test_costs(self):
        """
        Check if value of the objective function that comes with the solved samples is equal to the one calculated from
        the powers and cost coefficients.

        Returns
        -------
        """

        for loader in self.loaders:
            for batch in loader:
                error = calculate_costs_per_grid(batch, batch.y_dict) - batch.objective
                self.assertLess(error.abs().sum(), self.cfg.cost_tolerance)
