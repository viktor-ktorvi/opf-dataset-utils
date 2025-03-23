from unittest import TestCase

import torch
from omegaconf import DictConfig
from torch_geometric.loader import DataLoader

from opf_dataset_utils.admittance import calculate_admittance_matrix
from opf_dataset_utils.power import calculate_bus_powers
from opf_dataset_utils.voltage import get_voltage_angles, get_voltages_magnitudes
from tests.utils import setup_test


class TestAdmittanceMatrix(TestCase):
    cfg: DictConfig
    device: torch.device
    loaders: list[DataLoader]

    @classmethod
    def setUpClass(cls):
        """
        Initialize test. Load config and test data.

        Returns
        -------
        """
        setup_test(cls)

    def test_absolute_errors_less_than_tolerance(self):
        """
        Check if the admittance matrix had been calculated correctly by calculating the complex power injections in two
        different ways -- one with and without the admittance matrix. Compare the mismatches.

        Returns
        -------
        """

        for loader in self.loaders:
            for batch in loader:
                Y_bus, Y_f, Y_t = calculate_admittance_matrix(batch)

                Vm = get_voltages_magnitudes(batch.y_dict)
                Va = get_voltage_angles(batch.y_dict)
                V = Vm * torch.exp(1j * Va)
                S_bus = torch.diag(V) @ torch.conj(Y_bus @ V)

                S_bus_true = calculate_bus_powers(batch, batch.y_dict)

                self.assertLess(
                    (S_bus_true - S_bus).abs().max().item(),
                    self.cfg.power_flow_error_tolerance_pu,
                )
