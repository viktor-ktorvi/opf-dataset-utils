from unittest import TestCase

import torch
from omegaconf import DictConfig
from torch_geometric.loader import DataLoader

from opf_dataset_utils.enumerations import NodeTypes
from opf_dataset_utils.physics.errors.power_flow import calculate_power_flow_errors
from opf_dataset_utils.physics.metrics.power_flow import AbsolutePowerFlowError
from tests.utils import setup_test


class TestPowerFlowMetrics(TestCase):
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

    def test_absolute_power_flow_metrics(self):
        """
        Check if the values calculated by the metric classes align with calculating the same values explicitly.
        Returns
        -------

        """
        mean_apparent_pu_metric = AbsolutePowerFlowError(aggr="mean", power_type="apparent", unit="per-unit").to(self.device)
        max_active_mega_metric = AbsolutePowerFlowError(aggr="max", power_type="active", unit="mega").to(self.device)
        min_reactive_kilo_metric = AbsolutePowerFlowError(aggr="min", power_type="reactive", unit="kilo").to(self.device)

        for loader in self.loaders:
            error_values = []
            baseMVA_values = []
            for batch in loader:
                mean_apparent_pu_metric.update(batch, batch.y_dict)
                max_active_mega_metric.update(batch, batch.y_dict)
                min_reactive_kilo_metric.update(batch, batch.y_dict)

                error_values.append(calculate_power_flow_errors(batch, batch.y_dict).cpu())
                baseMVA_values.append(batch.x[batch.batch_dict[NodeTypes.BUS]].cpu())

            errors_pu = torch.cat(error_values)
            errors_mega = errors_pu * torch.cat(baseMVA_values)
            errors_kilo = errors_mega * 1e3

            self.assertAlmostEqual(mean_apparent_pu_metric.compute().cpu().item(), errors_pu.abs().mean().item())
            self.assertAlmostEqual(max_active_mega_metric.compute().cpu().item(), errors_mega.real.abs().max().item())
            self.assertAlmostEqual(min_reactive_kilo_metric.compute().cpu().item(), errors_kilo.imag.abs().min().item())

            mean_apparent_pu_metric.reset()
            max_active_mega_metric.reset()
            min_reactive_kilo_metric.reset()
