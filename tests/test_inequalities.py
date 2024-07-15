from typing import List
from unittest import TestCase

import torch
from omegaconf import DictConfig
from torch_geometric.loader import DataLoader

from opf_dataset_utils.enumerations import EdgeTypes
from opf_dataset_utils.physics.errors.inequality.branch_power import (
    calculate_branch_power_errors_from,
    calculate_branch_power_errors_to,
)
from opf_dataset_utils.physics.errors.inequality.generator_power import (
    calculate_lower_active_power_errors,
    calculate_lower_reactive_power_errors,
    calculate_upper_active_power_errors,
    calculate_upper_reactive_power_errors,
)
from opf_dataset_utils.physics.errors.inequality.voltage import (
    calculate_lower_voltage_angle_difference_errors,
    calculate_lower_voltage_magnitude_errors,
    calculate_upper_voltage_angle_difference_errors,
    calculate_upper_voltage_magnitude_errors,
)
from opf_dataset_utils.physics.voltage import get_reference_voltage_angles
from tests.utils import setup_test


class TestPowerFlow(TestCase):
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

    def test_voltage(self):
        """
        Test voltage magnitude and angle difference inequalities. Test reference voltage equality.
        Returns
        -------

        """
        for loader in self.loaders:
            for batch in loader:
                self.assertLess(
                    calculate_upper_voltage_magnitude_errors(batch, batch.y_dict).abs().max(),
                    self.cfg.voltage_magnitude_tolerance_pu,
                )
                self.assertLess(
                    calculate_lower_voltage_magnitude_errors(batch, batch.y_dict).abs().max(),
                    self.cfg.voltage_magnitude_tolerance_pu,
                )

                self.assertLess(
                    calculate_upper_voltage_angle_difference_errors(batch, batch.y_dict, EdgeTypes.TRANSFORMER)
                    .abs()
                    .max(),
                    self.cfg.voltage_angle_tolerance_rad,
                )
                self.assertLess(
                    calculate_upper_voltage_angle_difference_errors(batch, batch.y_dict, EdgeTypes.AC_LINE).abs().max(),
                    self.cfg.voltage_angle_tolerance_rad,
                )

                self.assertLess(
                    calculate_lower_voltage_angle_difference_errors(batch, batch.y_dict, EdgeTypes.TRANSFORMER)
                    .abs()
                    .max(),
                    self.cfg.voltage_angle_tolerance_rad,
                )
                self.assertLess(
                    calculate_lower_voltage_angle_difference_errors(batch, batch.y_dict, EdgeTypes.AC_LINE).abs().max(),
                    self.cfg.voltage_angle_tolerance_rad,
                )

                self.assertLess(
                    get_reference_voltage_angles(batch, batch.y_dict).abs().max(), self.cfg.voltage_angle_tolerance_rad
                )

    def test_generator_power(self):
        """
        Test generator power inequalities.
        Returns
        -------

        """
        for loader in self.loaders:
            for batch in loader:
                self.assertLess(
                    calculate_upper_active_power_errors(batch, batch.y_dict).abs().max(),
                    self.cfg.generator_power_tolerance_pu,
                )
                self.assertLess(
                    calculate_lower_active_power_errors(batch, batch.y_dict).abs().max(),
                    self.cfg.generator_power_tolerance_pu,
                )

                self.assertLess(
                    calculate_upper_reactive_power_errors(batch, batch.y_dict).abs().max(),
                    self.cfg.generator_power_tolerance_pu,
                )
                self.assertLess(
                    calculate_lower_reactive_power_errors(batch, batch.y_dict).abs().max(),
                    self.cfg.generator_power_tolerance_pu,
                )

    def test_branch_power(self):
        """
        Test branch power inequalities.
        Returns
        -------

        """
        for loader in self.loaders:
            for batch in loader:
                self.assertLess(
                    calculate_branch_power_errors_from(batch, batch.y_dict, EdgeTypes.TRANSFORMER).abs().max(),
                    self.cfg.branch_powers_error_tolerance_pu,
                )
                self.assertLess(
                    calculate_branch_power_errors_from(batch, batch.y_dict, EdgeTypes.AC_LINE).abs().max(),
                    self.cfg.branch_powers_error_tolerance_pu,
                )

                self.assertLess(
                    calculate_branch_power_errors_to(batch, batch.y_dict, EdgeTypes.TRANSFORMER).abs().max(),
                    self.cfg.branch_powers_error_tolerance_pu,
                )
                self.assertLess(
                    calculate_branch_power_errors_to(batch, batch.y_dict, EdgeTypes.AC_LINE).abs().max(),
                    self.cfg.branch_powers_error_tolerance_pu,
                )
