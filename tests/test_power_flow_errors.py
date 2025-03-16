from typing import List
from unittest import TestCase

import torch
from omegaconf import DictConfig
from torch_geometric.loader import DataLoader

from opf_dataset_utils.enumerations import (
    EdgeTypes,
    NodeTypes,
    SolutionACLineIndices,
    SolutionTransformerIndices,
)
from opf_dataset_utils.errors.power_flow import calculate_power_flow_errors
from opf_dataset_utils.power import calculate_branch_powers
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

    def test_absolute_errors_less_than_tolerance(self):
        """
        Check if the maximum module of the complex power flow errors [p.u.] is less than a tolerance.

        Returns
        -------
        """

        for loader in self.loaders:
            for batch in loader:
                self.assertLess(
                    calculate_power_flow_errors(batch, batch.y_dict).abs().max().item(),
                    self.cfg.power_flow_error_tolerance_pu,
                )

    def test_branch_flows_equal_solution(self):
        """
        Check if the maximum module of the difference between the calculated and solution branch power flows [p.u.] is
        less than a tolerance.

        Returns
        -------
        """
        for loader in self.loaders:
            for batch in loader:
                ac_line_powers_from, ac_line_powers_to = calculate_branch_powers(batch, batch.y_dict, EdgeTypes.AC_LINE)
                transformer_powers_from, transformer_powers_to = calculate_branch_powers(
                    batch, batch.y_dict, EdgeTypes.TRANSFORMER
                )

                ac_line_solutions = batch.edge_label_dict[(NodeTypes.BUS, EdgeTypes.AC_LINE, NodeTypes.BUS)]
                transformer_solutions = batch.edge_label_dict[(NodeTypes.BUS, EdgeTypes.TRANSFORMER, NodeTypes.BUS)]

                ac_line_powers_solution_from = (
                    ac_line_solutions[:, SolutionACLineIndices.ACTIVE_POWER_FROM]
                    + 1j * ac_line_solutions[:, SolutionACLineIndices.REACTIVE_POWER_FROM]
                )
                ac_line_powers_solution_to = (
                    ac_line_solutions[:, SolutionACLineIndices.ACTIVE_POWER_TO]
                    + 1j * ac_line_solutions[:, SolutionACLineIndices.REACTIVE_POWER_TO]
                )

                transformer_powers_solution_from = (
                    transformer_solutions[:, SolutionTransformerIndices.ACTIVE_POWER_FROM]
                    + 1j * transformer_solutions[:, SolutionTransformerIndices.REACTIVE_POWER_FROM]
                )
                transformer_powers_solution_to = (
                    transformer_solutions[:, SolutionTransformerIndices.ACTIVE_POWER_TO]
                    + 1j * transformer_solutions[:, SolutionTransformerIndices.REACTIVE_POWER_TO]
                )

                self.assertLess(
                    (ac_line_powers_from - ac_line_powers_solution_from).abs().max().item(),
                    self.cfg.branch_powers_error_tolerance_pu,
                )
                self.assertLess(
                    (ac_line_powers_to - ac_line_powers_solution_to).abs().max().item(),
                    self.cfg.branch_powers_error_tolerance_pu,
                )
                self.assertLess(
                    (transformer_powers_from - transformer_powers_solution_from).abs().max().item(),
                    self.cfg.branch_powers_error_tolerance_pu,
                )
                self.assertLess(
                    (transformer_powers_to - transformer_powers_solution_to).abs().max().item(),
                    self.cfg.branch_powers_error_tolerance_pu,
                )
