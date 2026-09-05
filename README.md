# CS5228 Group 6

This repository contains the project datasets, exploratory analysis, preprocessing,
and baseline modelling workflow.

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

Then open the notebooks:

- `notebooks/eda.ipynb` explores the raw datasets.
- `notebooks/data_preprocessing.ipynb` builds the model-ready files in `data/processed/`.
- `notebooks/auxiliary_data_investigation.ipynb` assesses the supplied external datasets.
- `notebooks/ridge_baseline.ipynb` evaluates the regularized linear baseline with
  chronological validation and produces in-memory test predictions.

The notebooks find the repository data whether JupyterLab is launched from the
repository root or from the `notebooks/` directory. Run the preprocessing notebook
from top to bottom before training a model.

## Generate the Ridge baseline submission

After `data/processed/train_preprocessed.csv`, `test_preprocessed.csv`, and
`preprocessing_metadata.json` have been generated, run from the repository root:

```bash
conda activate cs5228
python scripts/create_ridge_submission.py
```

The script reproduces the final model configuration selected in
`notebooks/ridge_baseline.ipynb`: core-only features, sparse one-hot encoding, and
Ridge regression with `alpha=10`. It loads `data/example-submission.csv`, maps test
predictions by `ROW_ID`, validates the required `Id,Predicted` schema and row order,
and writes:

```text
submissions/ridge_baseline.csv
```

Use `--output` to write elsewhere, or inspect all supported options with:

```bash
python scripts/create_ridge_submission.py --help
```

For example:

```bash
python scripts/create_ridge_submission.py \
  --output submissions/ridge_alpha_100.csv \
  --alpha 100
```

Changing model options creates an experiment rather than the documented baseline;
compare it using the chronological validation in the notebook before submission.
