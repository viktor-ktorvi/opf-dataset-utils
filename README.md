# opf-dataset-utils

In this package we provide utils to support working with the
[OPFDataset](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.OPFDataset.html#torch_geometric.datasets.OPFDataset).

We implement:

* Efficient OPF related calculations to be used as metrics or in a physics informed setting:
    * Power Flow errors
    * Branch powers
    * Costs
    * Inequalities
* Data visualization
* Enums for indexing the OPFData JSON format
* And more...

## Installation

Installation is currently available directly from GitHub:

```
pip install git+https://github.com/viktor-ktorvi/opf-dataset-utils.git
```

## Usage

### Plotting

See [scripts/draw.py](scripts/draw.py) for a full example.

```python
from opf_dataset_utils.plotting.draw import draw_graph

draw_graph(dataset[0], ax=ax, node_size=300)
```

<p align="center">
<img src="img/draw_example.png" alt="Example graph" width="600"/>
</p>

### OPF calculations

#### Power flow errors

See [scripts/power_flow_errors.py](scripts/power_flow_errors.py) for a full example.

```python
from opf_dataset_utils.physics.errors.power_flow import calculate_power_flow_errors

print("Mean power flow errors:")
print(f"\tSolution: {calculate_power_flow_errors(batch, batch.y_dict).abs().mean():.5e}")
print(f"\tUntrained model prediction: {calculate_power_flow_errors(batch, predictions).abs().mean():.5f}")
```

```
Mean power flow errors:
	Solution: 1.28563e-06 [p.u.]
	Untrained model prediction: 413350.84375 [p.u.]
```

#### Costs

See [scripts/costs.py](scripts/costs.py) for a full example.

```python
from opf_dataset_utils.costs import (
    calculate_costs_per_generator,
    calculate_costs_per_grid,
)

costs_per_grid = calculate_costs_per_grid(data, data.y_dict)
costs_per_generator = calculate_costs_per_generator(data, data.y_dict)
```

#### Inequality violations

See [scripts/inequality_errors.py](scripts/inequality_errors.py) for a full example.

```python
from opf_dataset_utils.enumerations import EdgeTypes
from opf_dataset_utils.physics.errors.inequality.voltage import calculate_upper_voltage_angle_difference_errors
from opf_dataset_utils.physics.errors.inequality.generator_power import calculate_lower_active_power_errors

upper_voltage_angle_violations_transformer = calculate_upper_voltage_angle_difference_errors(data, data.y_dict, EdgeTypes.TRANSFORMER)
lower_active_power_generation_violations = calculate_lower_active_power_errors(data, data.y_dict)
# etc.
```

#### Branch power flows

See [scripts/branch_powers.py](scripts/branch_powers.py) for a full example.

```python
from opf_dataset_utils.enumerations import EdgeTypes
from opf_dataset_utils.physics.power import calculate_branch_powers

ac_line_powers_from, ac_line_powers_to = calculate_branch_powers(batch, batch.y_dict, EdgeTypes.AC_LINE)
transformer_powers_from, transformer_powers_to = calculate_branch_powers(batch, batch.y_dict, EdgeTypes.TRANSFORMER)
```

#### Etc.
