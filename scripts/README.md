# Scripts

This folder is dedicated to scripts, whether python, bash or sbatch.

Generally, scripts are more for standalone processes while figuring them out.

Once a script is more mature, it should be generalized and integrated into the package
itself.

## Scripts

| Scripts                | Description                                                                                                                        |
|------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| `branch_powers.py`     | Calculate branch powers. : `python3 -m scripts.branch_powers`                                                                      |
| `costs.py`             | Calculate costs per generator and per grid. : `python3 -m scripts.costs`                                                           |
| `draw.py`              | Draw the graph of a sample from the OPFDataset. : `python3 -m scripts.draw`                                                        |
| `power_flow_errors.py` | Calculate power flow errors of a solved instance and one predicted by an untrained model. : `python3 -m scripts.power_flow_errors` |
| `inequality_errors.py` | Calculate the various inequality violations of the solution and an untrained model. : `python3 -m scripts.inequality_errors`       |        
