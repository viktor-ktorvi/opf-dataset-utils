import os
from pathlib import Path

import hydra
from omegaconf import DictConfig

from opf_dataset_utils import CONFIG_PATH
from scripts.experiments.utils.sweep import create_sweep_sbatch_script


@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="experiments")
def main(cfg: DictConfig):
    """
    Run a number of wandb sweep agents using an sbatch script. If a script path isn't specified, a new one will be
    created as defined in the project config.

    Parameters
    ----------
    cfg: DictConfig
        Config.

    Returns
    -------
    """
    if cfg.slurm.existing_sbatch_script_path is None:
        create_sweep_sbatch_script(cfg)
        sbatch_script_path = Path(cfg.slurm.path) / cfg.slurm.sbatch_script_name
    else:
        sbatch_script_path = cfg.slurm.existing_sbatch_script_path

    for i in range(cfg.slurm.num_agents):
        os.system(f"sbatch {sbatch_script_path}")


if __name__ == "__main__":
    main()
