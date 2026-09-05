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

## Run the notebooks

With the `cs5228` environment active, start JupyterLab:

```bash
jupyter lab
```

Then open the notebooks:

- `notebooks/01_eda.ipynb` explores the raw datasets.
- `notebooks/02_auxiliary_data_investigation.ipynb` assesses the supplied external datasets.
- `notebooks/03_data_preprocessing.ipynb` removes raw and normalization-induced
  exact duplicates, audits feature duplicates, and builds the core model-ready
  files in `data/processed/core/`.
- `notebooks/04_auxiliary_data_preprocessing.ipynb` adds HDB features and a
  conservative pre-2021 MRT-network proxy, then collapses duplicate training
  feature vectors by mean rent and writes the result to `data/processed/hdb_mrt/`.
- `notebooks/05_ridge_baseline.ipynb` evaluates both the core and core + HDB/MRT
  variants using expanding chronological validation. Fold-specific preprocessing
  learns rare categories only from each fold's training months and writes two test
  prediction files.

The notebooks find the repository data whether JupyterLab is launched from the
repository root or from the `notebooks/` directory. Run the preprocessing notebooks
from top to bottom before training a model. Run `03_data_preprocessing.ipynb` first,
then `04_auxiliary_data_preprocessing.ipynb`.

## Processed data layout

```text
data/processed/
├── core/
│   ├── train_preprocessed.csv
│   ├── test_preprocessed.csv
│   └── preprocessing_metadata.json
├── hdb_mrt/
│   ├── train_preprocessed.csv
│   ├── test_preprocessed.csv
│   └── preprocessing_metadata.json
└── auxiliary/
    ├── hdb_spatial_features.csv
    ├── market_features_lag1.csv
    ├── train_hdb_mrt_features.csv
    └── test_hdb_mrt_features.csv
```

## Ridge baseline submissions

Running `notebooks/05_ridge_baseline.ipynb` writes:

```text
submissions/ridge_baseline_core.csv
submissions/ridge_baseline_hdb_mrt.csv
```

## Generate a Ridge submission from the command line

After the files in `data/processed/hdb_mrt/` have been generated, run from the
repository root:

```bash
conda activate cs5228
python scripts/create_ridge_submission.py
```

The script reads its model and categorical feature lists from
`data/processed/hdb_mrt/preprocessing_metadata.json`, applies sparse one-hot
encoding, and fits Ridge regression with `alpha=10`. It loads
`data/example-submission.csv`, maps test predictions by `ROW_ID`, validates the
required `Id,Predicted` schema and row order, and writes:

```text
submissions/ridge_baseline_hdb_mrt.csv
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
