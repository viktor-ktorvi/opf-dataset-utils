from pathlib import Path

import yaml
from omegaconf import DictConfig, OmegaConf

import wandb

# TODO add docs on how to run
#  like another readme in the experiments


def convert_config_to_wandb_format(parameters_dict: dict) -> dict:
    """
    Convert a dictionary of parameters into the correct wandb sweep config format by adding 'parameters' and 'values'
    into the hierarchy, where appropriate.

    Parameters
    ----------
    parameters_dict: dict
        Dictionary that defines a hyperparameter sweep.
    Returns
    -------
    converted_parameters_dict: dict
        A dictionary of parameters in the correct wandb sweep config format.

    Raises
    ------
    ValueError:
        If a value field of the dictionary is neither a dictionary nor a list.
    """
    for key in list(parameters_dict.keys()):
        # if the field is empty, remove it
        if not parameters_dict[key]:
            del parameters_dict[key]
            continue

        if isinstance(parameters_dict[key], dict):
            parameters_dict[key] = {"parameters": convert_config_to_wandb_format(parameters_dict[key])}
        elif isinstance(parameters_dict[key], list):
            parameters_dict[key] = {"values": parameters_dict[key]}
        else:
            raise ValueError(
                f"A value of the parameters dictionary must be either a dictionary or a list. Got {type(parameters_dict[key])} instead."
            )

    return parameters_dict


def create_sweep_config(cfg: DictConfig):
    """
    Create a wandb sweep config.

    Parameters
    ----------
    cfg: DictConfig
        Project configuration. Defines the sweep config, among other things.

    Returns
    -------
    sweep_config: Dict
        Sweep config dictionary.
    """
    cfg_parameters = OmegaConf.to_container(cfg.sweep.parameters, resolve=True, throw_on_missing=True)
    parameters_dict = convert_config_to_wandb_format(cfg_parameters)

    sweep_config = dict(
        project=cfg.wandb.project,
        method=cfg.sweep.method,
        name=cfg.sweep.name,
        program=cfg.sweep.program,
        parameters=parameters_dict,
        command=[
            "${env}",
            "${interpreter}",
            "-m",
            "${program}",
        ]
        + list(cfg.sweep.overrides),
    )

    return sweep_config


def create_sweep_sbatch_script(cfg: DictConfig):
    """
    Generates an sbatch script for a wandb sweep. If a sweep ID is not provided, a sweep will be started either using
    the sweep config path if it itself is provided, otherwise, a sweep config will be created using the project config.

    Parameters
    ----------
    cfg: DictConfig
        Config.

    Returns
    -------
    """
    # get sweep id
    if cfg.slurm.existing_sweep_id is not None:
        # sweep already exists
        sweep_id = cfg.slurm.existing_sweep_id
    else:
        if cfg.slurm.existing_sweep_config_path is not None:
            # sweep config already exists
            with open(cfg.slurm.existing_sweep_config_path, "r") as f:
                sweep_config = yaml.load(f, Loader=yaml.SafeLoader)
        else:
            sweep_config = create_sweep_config(cfg)

        sweep_id = wandb.sweep(sweep=sweep_config, project=cfg.wandb.project)

    # create sbatch script
    SBATCH_dashdash = "#SBATCH --"

    lines = (
        f"#!/bin/bash\n" f"{SBATCH_dashdash}job-name={cfg.slurm.job_name}",
        f"{SBATCH_dashdash}output={cfg.slurm.output}",
        f"{SBATCH_dashdash}error={cfg.slurm.error}",
        f"{SBATCH_dashdash}ntasks={cfg.slurm.ntasks}",
        f"{SBATCH_dashdash}cpus-per-task={cfg.slurm.cpus_per_task}",
        f"{SBATCH_dashdash}time={cfg.slurm.time}",
        f"{SBATCH_dashdash}mem={cfg.slurm.mem}",
        f"{SBATCH_dashdash}gres={cfg.slurm.gres}",
        '\nexport WANDB_DISABLE_SERVICE="True"',
        "module load miniconda/3",
        "conda activate opf-dataset-utils-env",
        f"wandb agent {cfg.wandb.project}/{sweep_id}",
    )

    Path(cfg.slurm.path).mkdir(exist_ok=True, parents=True)

    with open(Path(cfg.slurm.path) / cfg.slurm.sbatch_script_name, "w") as f:
        for line in lines:
            f.write(f"{line}\n")
