# Scripts

This folder is dedicated to scripts, whether python, bash or sbatch.

Generally, scripts are more for standalone processes while figuring them out.

Once a script is more mature, it should be generalized and integrated into the package
itself.

## Scripts

| Scripts                | Description                                                                                                                        |
|------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| `power_flow_errors.py` | Calculate power flow errors of a solved instance and one predicted by an untrained model. : `python3 -m scripts.power_flow_errors` |
| `draw.py`              | Draw the graph of a sample from the OPFDataset. : `python3 -m scripts.draw`                                                        |
| `branch_powers.py`     | Calculate branch powers. : `python3 -m scripts.branch_powers`                                                                      |
