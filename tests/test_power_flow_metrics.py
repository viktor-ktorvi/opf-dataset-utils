from unittest import TestCase

import torch
from omegaconf import DictConfig
from torch_geometric.loader import DataLoader

from opf_dataset_utils.enumerations import NodeTypes
from opf_dataset_utils.errors.power_flow import calculate_power_flow_errors
from opf_dataset_utils.metrics.power_flow import PowerFlowError
from opf_dataset_utils.power import calculate_bus_powers
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
        mean_apparent_pu_metric = PowerFlowError(
            aggr="mean", power_type="apparent", value_type="absolute", unit="per-unit"
        ).to(self.device)
        max_active_mega_metric = PowerFlowError(aggr="max", power_type="active", value_type="absolute", unit="mega").to(
            self.device
        )
        min_reactive_kilo_metric = PowerFlowError(
            aggr="min", power_type="reactive", value_type="absolute", unit="kilo"
        ).to(self.device)

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

    def test_relative_power_flow_metrics(self):
        """
        Check if the values calculated by the metric classes align with calculating the same values explicitly.
        Returns
        -------

        """
        mean_apparent_metric = PowerFlowError(aggr="mean", power_type="apparent", value_type="relative").to(self.device)
        max_active_metric = PowerFlowError(aggr="max", power_type="active", value_type="relative").to(self.device)
        min_reactive_metric = PowerFlowError(aggr="min", power_type="reactive", value_type="relative").to(self.device)

        for loader in self.loaders:
            abs_error_values = []
            power_values = []

            for batch in loader:
                mean_apparent_metric.update(batch, batch.y_dict)
                max_active_metric.update(batch, batch.y_dict)
                min_reactive_metric.update(batch, batch.y_dict)

                abs_error_values.append(calculate_power_flow_errors(batch, batch.y_dict).cpu())
                power_values.append(calculate_bus_powers(batch, batch.y_dict).cpu())

            errors_pu = torch.cat(abs_error_values)
            powers_pu = torch.cat(power_values)

            # apparent
            errors_apparent = errors_pu.abs()
            powers_apparent = powers_pu.abs()
            mask_apparent = powers_apparent > 0.0
            relative_apparent = errors_apparent[mask_apparent] / powers_apparent[mask_apparent] * 100

            # active
            errors_real_abs = errors_pu.real.abs()
            powers_real_abs = powers_pu.real.abs()
            mask_real = powers_real_abs > 0.0
            relative_real = errors_real_abs[mask_real] / powers_real_abs[mask_real] * 100

            # reactive
            errors_imag_abs = errors_pu.imag.abs()
            powers_imag_abs = powers_pu.imag.abs()
            mask_imag = powers_imag_abs > 0.0
            relative_imag = errors_imag_abs[mask_imag] / powers_imag_abs[mask_imag] * 100

            self.assertAlmostEqual(mean_apparent_metric.compute().cpu().item(), relative_apparent.mean().item())
            self.assertAlmostEqual(max_active_metric.compute().cpu().item(), relative_real.max().item())
            self.assertAlmostEqual(min_reactive_metric.compute().cpu().item(), relative_imag.min().item())

            mean_apparent_metric.reset()
            max_active_metric.reset()
            min_reactive_metric.reset()
