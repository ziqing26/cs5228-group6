# CS5228 Group 6

This repository contains the project datasets and exploratory data analysis notebook.

## Environment setup

Install [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) or Anaconda, then create the project environment from the repository root:

```bash
conda env create -f environment.yml
conda activate cs5228
```

To update an existing environment after `environment.yml` changes:

```bash
conda env update -f environment.yml --prune
```

## Run the notebook

With the `cs5228` environment active, start JupyterLab:

```bash
jupyter lab
```

Then open `notebooks/eda.ipynb`. The notebook finds `data/train.csv` whether JupyterLab is launched from the repository root or from the `notebooks/` directory.

