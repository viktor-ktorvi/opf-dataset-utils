# Generating diverse power grid topologies via subgraph sampling

To reproduce the results from the _Generating diverse power grid topologies via subgraph
sampling_ paper, follow the instructions below. Please raise an issue if errors occur.

## Installation

Clone the repository:

```
git clone -b subgraph-sampling git@github.com:viktor-ktorvi/opf-dataset-utils.git
```

Make sure make is installed:
```
sudo apt-get update
sudo apt install build-essential
```

Install the conda environment
```
cd opf-dataset-utils
make conda-create-env
conda activate building-sci-env
make install
```

## Reproducing the results

Run the following scripts from the repository root.

| Scripts                                            | Description                                                                                                                                                                                           |
|----------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `scripts/sample_subgraphs_different_parameters.sh` | Sample example topologies with different hyperparameters and visualize them. The results will be saved as PDF files in `<REPO_ROOT/img/>`: `source scripts/sample_subgraphs_different_parameters.sh ` |
| `scripts/edit_distances.sh`                        | Approximate the (within-dataset) edit distances for different datasets. : `source scripts/edit_distances.sh`                                                                                          |