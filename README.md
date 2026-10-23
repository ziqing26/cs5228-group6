# CS5228 Group 6 Ox Alpha

This repository contains the project datasets, exploratory analysis, preprocessing,
and chronological model comparisons for Singapore HDB rental prediction.
The main findings are in `reports/final_report_group6.tex` and its compiled PDF.

## Environment setup

Install [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) or Anaconda, then create the project environment from the repository root:

```bash
conda env create -f environment.yml
conda activate cs5228
```

Register the active environment for the automated notebook runner:

```bash
python -m ipykernel install --sys-prefix --name python3 --display-name "CS5228"
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
  variants on identical individual core rows using expanding chronological
  validation. It joins uncollapsed auxiliary features by `ROW_ID` and `RENT_DATE`,
  preserving original targets for training and evaluation. Each fold learns its
  own preprocessing and rare categories. The notebook writes two prediction files.

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

## Improvement experiments

Notebooks 01–05 are preserved. Run the new notebooks in order:

- `06_preprocessing_audit.ipynb` compares training duplicate policies and
  fold-fitted block recovery, with identical individual validation records.
- `07_feature_experiments.ipynb` evaluates Ridge with schools, malls, and
  strictly historical rental features.
- `08_catboost_model.ipynb` evaluates CatBoost and the same feature ablations.
  It searches all 20 combinations of depths `[6, 8]`, learning rates
  `[0.001, 0.005, 0.01, 0.05, 0.08]`, and L2 regularization `[3, 5]`
  on the two chronological folds before testing additional features.
- `09_mlp_model.ipynb` evaluates an MLP with chronological epoch selection and
  three seeds for stability.
- `10_final_evaluation.ipynb` freezes the earlier-fold winner, evaluates the
  latest block and a longer horizon, and writes a separate improved submission.

Update the environment using the setup command above before running these
notebooks. Shared implementation lives in `experiments/rental_experiments.py`;
experiment scores, policy decisions, and diagnostics go to `experiments/results/`.
Existing processed data and baseline submissions are not overwritten.

To execute the new notebooks from the repository root:

```bash
python scripts/run_improvement_notebooks.py
```

Use `--start 8 --end 9` to rerun a subset after its prerequisite results exist.
The experiment plan is in `NEXT_STEPS.md`; findings and limitations are documented
in the LaTeX report. A local summary and its writer live in the ignored
`artefacts/` directory; neither is required to reproduce experiments.

## Ridge baseline submissions

Running `notebooks/05_ridge_baseline.ipynb` writes:

```text
submissions/ridge_baseline_core.csv
submissions/ridge_baseline_hdb_mrt.csv
```

## Generate a Ridge submission from the command line

The script below still uses the collapsed HDB/MRT training file. Run the notebook
to reproduce the fair comparison on individual rows. Running this script afterward
overwrites the auxiliary submission with the legacy variant's predictions.

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

## Reproduce the report results

Run every command from the repository root. Required inputs are
`data/train.csv`, `data/test.csv`, `data/example-submission.csv`, and the six
supplied CSVs in `data/auxiliary/`. The environment in `environment.yml` is a
supported setup, but uses newer Python/NumPy versions than the recorded run.
For the closest reproduction of the reported scores, create a separate environment
matching `experiments/results/final_refit.json`:

```bash
conda create -n cs5228-report -c conda-forge python=3.11.5 numpy=1.26.4 pandas=2.1.4 scikit-learn=1.2.2 matplotlib=3 jupyterlab=4 ipykernel=6 nbformat=5 nbclient pip
conda activate cs5228-report
python -m pip install catboost==1.2.10
python -m ipykernel install --sys-prefix --name python3 --display-name "CS5228 report"
```

Register the active environment's `python3` kernel for either setup: the notebook
runner explicitly uses that kernel name. Exact package versions improve
comparability; CPU/platform differences can still cause numerical or runtime
variation. Seeds, feature contracts and input/implementation hashes are recorded
with the results.

1. Execute exploratory and preprocessing notebooks in order to inspect the
   source data and build the prerequisite auxiliary files:

   ```bash
   jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/01_eda.ipynb notebooks/02_auxiliary_data_investigation.ipynb notebooks/03_data_preprocessing.ipynb notebooks/04_auxiliary_data_preprocessing.ipynb
   ```

2. Run the controlled preprocessing audit, Ridge/feature comparison, CatBoost
   grid, MLP comparison and final evaluation:

   ```bash
   python scripts/run_improvement_notebooks.py
   ```

   This overwrites the executed notebooks and experiment outputs, including
   `submissions/catboost_improved.csv`. The original Ridge baseline notebook is
   optional for the final report comparison; run it separately to reproduce the
   legacy reference. Its deduplicated validation population differs from the
   final report's original-row comparisons.

3. Regenerate the two EDA figures and compile the report:

   ```bash
   python scripts/create_report_figures.py
   cd reports
   pdflatex -interaction=nonstopmode -halt-on-error final_report_group6.tex
   pdflatex -interaction=nonstopmode -halt-on-error final_report_group6.tex
   cd ..
   ```

   Install a TeX distribution providing pdfLaTeX, or use Overleaf with the entire
   `reports/` directory, including `IEEEtran.cls` and `figures/`. Alternatively,
   run `tectonic reports/final_report_group6.tex`; Tectonic resolves references
   automatically. LaTeX uses the figures relative to the source directory.
   Experiment execution regenerates data and scores; the report's prose and
   tables are maintained manually and should be checked against those outputs.

## Results and evidence in the report

- **Data and EDA:** raw CSVs and exploratory notebooks establish 150,000 training
  and 50,000 test records, the rent trend, category inconsistencies and outliers.
  `scripts/create_report_figures.py` exports the temporal and flat-type plots.
- **Preprocessing:** `data_audit.json`, `preprocessing_scores.csv`,
  `preprocessing_selection.json`, `invalid_values.csv` and
  `block_recovery_audit.csv` in `experiments/results/` document retention of all
  rows and fold-fitted invalid-source masking/recovery.
- **Model and feature selection:** `ridge_scores.csv`, `catboost_scores.csv`,
  `mlp_scores.csv` and the corresponding selection JSONs document the two
  earlier windows. CatBoost uses 35 base features, depth 8, learning rate 0.05
  and leaf regularisation 3. History improves its mean RMSE by only 0.889 SGD,
  below the 1 SGD feature-retention threshold.
- **Results and errors:** `final_scores.csv`, `final_selection.json`, saved
  final predictions and the `*_errors_by_*.csv` files support the main tables
  and subgroup discussion. Later-block CatBoost RMSE is 458.740 SGD versus
  Ridge's 582.264 SGD (21.21% reduction); MAE is 334.412 SGD and
  actual-minus-predicted bias is 13.551 SGD. The 16-month diagnostic RMSE is
  534.106 SGD. The longer diagnostic overlaps the other windows and the later
  block was previously inspected; neither is an independent untouched test.
- **Submission:** `final_refit.json` and
  `data/processed/improved/preprocessing_metadata.json` record 499 trees,
  35 predictors and all 150,000 training rows. The submission contains 50,000
  predictions in sample ID order. No verified competition score is available.

The generated `data/processed/improved/` files are full-training prediction
artifacts. Cross-validation must rebuild transformations inside each training
fold through the shared implementation; using these full-data files for
validation would not reproduce the reported protocol.

`artefacts/write_report.py` and `artefacts/improvement_report.md` are local,
git-ignored working files. When present, regenerate the optional summary with
`python artefacts/write_report.py`. A clean checkout does not contain them and
does not need them to execute the final-evaluation notebook.

## Report consistency notes

The main text agrees on the selected CatBoost configuration, the 35-input
feature set, retained rows, original evaluation targets and signed-bias
convention. Preprocessing rules belong in the EDA section; model and feature
selection are described in Methods; split dates and leakage controls are
specified in Experimental Setup; Results interprets their measured effects.
These sections refer to the same procedure rather than separate experiments.

Remaining qualifications to keep visible before submission:

- The original Ridge baseline uses a deduplicated validation population;
  its scores must stay separate from the final report's common-population table.
- The long-horizon diagnostic overlaps the selection and later blocks, and the
  later block was previously inspected. No independent competition score or
  statistically significant improvement is claimed.
- Core/HDB/MRT inputs are selected as a combined base: no core-only CatBoost
  removal experiment isolates the contribution of HDB/MRT features.
- CatBoost's larger search and the MLP's small search support conclusions about
  the evaluated configurations, rather than general model-family superiority.
