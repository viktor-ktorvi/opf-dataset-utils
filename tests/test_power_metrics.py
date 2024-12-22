from unittest import TestCase

import torch
from omegaconf import DictConfig
from torch_geometric.loader import DataLoader

from opf_dataset_utils.enumerations import NodeTypes
from opf_dataset_utils.metrics.power import Power
from opf_dataset_utils.power import calculate_bus_powers
from tests.utils import setup_test


class TestPowerMetrics(TestCase):
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

    def test_power_metrics(self):
        """
        Check if the values calculated by the metric classes align with calculating the same values explicitly.
        Returns
        -------

        """
        mean_apparent_pu_metric = Power(aggr="mean", power_type="apparent", unit="per-unit").to(self.device)
        max_active_mega_metric = Power(aggr="max", power_type="active", unit="mega").to(self.device)
        min_reactive_kilo_metric = Power(aggr="min", power_type="reactive", unit="kilo").to(self.device)

        for loader in self.loaders:
            power_values = []
            baseMVA_values = []
            for batch in loader:
                mean_apparent_pu_metric.update(batch, batch.y_dict)
                max_active_mega_metric.update(batch, batch.y_dict)
                min_reactive_kilo_metric.update(batch, batch.y_dict)

                power_values.append(calculate_bus_powers(batch, batch.y_dict).cpu())
                baseMVA_values.append(batch.x[batch.batch_dict[NodeTypes.BUS]].cpu())

            powers_pu = torch.cat(power_values)
            powers_mega = powers_pu * torch.cat(baseMVA_values)
            powers_kilo = powers_mega * 1e3

            self.assertAlmostEqual(
                mean_apparent_pu_metric.compute().cpu().item(), powers_pu.abs().mean().item(), places=6
            )
            self.assertAlmostEqual(
                max_active_mega_metric.compute().cpu().item(), powers_mega.real.max().item(), places=6
            )
            self.assertAlmostEqual(
                min_reactive_kilo_metric.compute().cpu().item(), powers_kilo.imag.min().item(), places=6
            )

            mean_apparent_pu_metric.reset()
            max_active_mega_metric.reset()
            min_reactive_kilo_metric.reset()
