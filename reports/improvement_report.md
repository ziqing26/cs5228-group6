# Rental baseline improvement report

The earlier-fold selection chose **catboost** with **history** features. On the later diagnostic block its RMSE is **456.445 SGD**, compared with **582.264 SGD** for the new Ridge reference (21.61% RMSE reduction). This comparison uses original individual rent targets and the same validation observations.

## Scope and reproducibility

This report documents the new experiments in notebooks 06–10. Notebooks 01–05, original datasets, and baseline submissions were preserved. The question is whether preprocessing, nonlinear models, and additional amenity/history features improve future rental prediction without changing the evaluation targets.

Run `python scripts/run_improvement_notebooks.py` from the repository root after installing `environment.yml`. Shared code is in `experiments/rental_experiments.py`. Machine-readable scores, selections, diagnostics, and source hashes are in `experiments/results/`.

## Data and preprocessing audit

There are 150,000 raw training observations and 50,000 test observations. Raw training has 0 missing cells. There are 425 raw exact matches and 641 exact matches after deterministic normalization. Matching records cannot be confirmed as repeated transactions because actual flat identifiers are unavailable.

There are 21,974 repeated feature rows; the largest group has 10 observations, and 17,900 feature groups contain different rents. Averaging these targets changes the training objective and weights. It is evaluated only on the training portion, never by smoothing validation targets.

- collapse_mean, block_recovery: early-fold mean RMSE **603.962 SGD**.
- collapse_mean, median: early-fold mean RMSE **603.965 SGD**.
- exact_only, block_recovery: early-fold mean RMSE **600.792 SGD**.
- exact_only, median: early-fold mean RMSE **600.798 SGD**.
- keep_all, block_recovery: early-fold mean RMSE **600.808 SGD**.
- keep_all, median: early-fold mean RMSE **600.803 SGD**.

Selected policy: **keep_all**, with **block_recovery**. Within 1 SGD of best early-fold mean RMSE, prefer keep_all then exact_only; prefer masking/recovery when within 0.1 SGD of best imputation. These small imputation differences should not be interpreted as evidence of a substantial predictive improvement.

### Invalid values and recovery

A lease age below 0 or above 100 years is invalid. The legacy pipeline masked the age but left the source lease year exposed. The repair variant also masks the invalid source year. HDB maximum floors/completion years at or below zero are invalid; completion years producing negative or greater-than-100-year ages are also masked in the repair variant. Numeric parsing failures remain missing. These rules are heuristics, not proof of the correct underlying value.

- train, LEASE_COMMENCE_DATE: 1 invalid/missing rows across 1 addresses.
- train, HDB_MAX_FLOOR: 159 invalid/missing rows across 11 addresses.
- train, HDB_YEAR_COMPLETED: 159 invalid/missing rows across 11 addresses.
- train, LEASE_AGE_AT_RENT: 1 invalid/missing rows across 1 addresses.
- train, HDB_BUILDING_AGE_AT_RENT: 159 invalid/missing rows across 11 addresses.
- test, LEASE_COMMENCE_DATE: 0 invalid/missing rows across 0 addresses.
- test, HDB_MAX_FLOOR: 0 invalid/missing rows across 0 addresses.
- test, HDB_YEAR_COMPLETED: 0 invalid/missing rows across 0 addresses.
- test, LEASE_AGE_AT_RENT: 0 invalid/missing rows across 0 addresses.
- test, HDB_BUILDING_AGE_AT_RENT: 0 invalid/missing rows across 0 addresses.

Recovery uses normalized town/block/street keys and requires all valid training values at that address to agree. Original invalid flags are retained. Derived ages are recomputed; remaining gaps use training-fitted medians for Ridge/MLP or native numeric missing-value handling for CatBoost.

Across the two earlier validation blocks, 0 source values were recovered. HDB attributes come from a shared block lookup, so an absent value for a block cannot usually be reconstructed from another transaction at that same block. Full audit details are in `experiments/results/block_recovery_audit.csv`.

## Evaluation protocol

- Selection windows: July–December 2023 and January–June 2024. Each model trains on strictly earlier months.
- Final diagnostic block: July 2024–March 2025. It was already examined in the original baseline, so it is not completely untouched.
- Every new experiment evaluates the same raw individual observations and original rent targets. Scores are not directly interchangeable with Notebook 05 scores, which use exact-deduplicated validation records.
- Preprocessing, repair mappings, encodings, and target scaling are training-fitted. Tree counts/MLP epochs use the last three months inside each training portion, followed by refitting on the complete training portion.
- MLP uses seed 42 for selection; seeds 123 and 2026 assess stability. No best-seed selection is performed.
- Mean RMSE is an unweighted mean across the two selection blocks. Final and long-horizon metrics are pooled within each respective block.
- The final model is frozen using earlier-fold results before final-block evaluation. Final scores do not trigger winner switching.

## Model and feature experiments

Ridge tests alpha 10 and 100. CatBoost tests two configurations: depth 6, learning rate 0.08, regularization 3; and depth 8, learning rate 0.05, regularization 5. Its budget is 500 trees with 40-round internal early stopping. CatBoost respects chronological input order and accepts categorical features directly. MLP uses 64 → 32 units, sparse one-hot inputs, standardized numeric inputs and rent targets, batch size 1024, regularization 0.01, and an internal selection budget of 40 epochs with six stale epochs.

Ridge and CatBoost test schools, malls, both, and historical rental means/counts after choosing parameters on base features. History features use strictly earlier months for training rows and fixed training-cutoff summaries for evaluation rows. They summarize town × flat type and property, with training-only missing-value handling for unseen groups. MLP uses the original 35 features for a controlled model comparison.


### ridge

- base, `{"alpha": 100}`: mean RMSE **603.382 SGD**.
- base, `{"alpha": 10}`: mean RMSE **600.808 SGD**.
- history, `{"alpha": 10}`: mean RMSE **603.998 SGD**.
- malls, `{"alpha": 10}`: mean RMSE **600.853 SGD**.
- schools, `{"alpha": 10}`: mean RMSE **600.887 SGD**.
- schools_malls, `{"alpha": 10}`: mean RMSE **600.885 SGD**.

Selected feature group: **base**. Selected parameters: `{"alpha": 10}`. Base features are retained if an added group improves mean RMSE by less than 1 SGD.

### catboost

- base, `{"depth": 6, "l2_leaf_reg": 3, "learning_rate": 0.08}`: mean RMSE **549.123 SGD**.
- base, `{"depth": 8, "l2_leaf_reg": 5, "learning_rate": 0.05}`: mean RMSE **548.042 SGD**.
- history, `{"depth": 8, "l2_leaf_reg": 5, "learning_rate": 0.05}`: mean RMSE **546.241 SGD**.
- malls, `{"depth": 8, "l2_leaf_reg": 5, "learning_rate": 0.05}`: mean RMSE **547.455 SGD**.
- schools, `{"depth": 8, "l2_leaf_reg": 5, "learning_rate": 0.05}`: mean RMSE **547.651 SGD**.
- schools_malls, `{"depth": 8, "l2_leaf_reg": 5, "learning_rate": 0.05}`: mean RMSE **547.096 SGD**.

Selected feature group: **history**. Selected parameters: `{"depth": 8, "l2_leaf_reg": 5, "learning_rate": 0.05}`. Base features are retained if an added group improves mean RMSE by less than 1 SGD.

With the selected tree configuration, base-feature RMSE is 548.042 SGD and the selected feature-group RMSE is 546.241 SGD. The extra feature gain is 1.801 SGD; most of the improvement over Ridge comes from the model change.

### mlp

- base, `{"random_state": 123}`: mean RMSE **631.349 SGD**.
- base, `{"random_state": 2026}`: mean RMSE **639.407 SGD**.
- base, `{"random_state": 42}`: mean RMSE **609.986 SGD**.

Selected feature group: **base**. Selected parameters: `{"random_state": 42}`. Base features are retained if an added group improves mean RMSE by less than 1 SGD.

## Final diagnostics and selection

- ridge, final_block: RMSE **582.264 SGD**, MAE **432.274 SGD**, actual-minus-predicted bias **-337.793 SGD** (25,984 observations).
- catboost, final_block: RMSE **456.445 SGD**, MAE **334.165 SGD**, actual-minus-predicted bias **24.746 SGD** (25,984 observations).
- mlp, final_block: RMSE **469.729 SGD**, MAE **347.440 SGD**, actual-minus-predicted bias **34.853 SGD** (25,984 observations).
- ridge, 16_month_backtest: RMSE **624.419 SGD**, MAE **473.995 SGD**, actual-minus-predicted bias **-263.517 SGD** (47,528 observations).
- catboost, 16_month_backtest: RMSE **532.926 SGD**, MAE **408.465 SGD**, actual-minus-predicted bias **18.972 SGD** (47,528 observations).

The earlier-fold winner is **catboost** with **history** features. Its selected early-fold mean RMSE is **546.241 SGD**. The separate submission is `submissions/catboost_improved.csv` and contains 50,000 finite predictions in sample-submission ID order.

The longer diagnostic predicts July 2023–October 2024 using data before July 2023. It tests a 16-month horizon and overlaps selection and final blocks; it is an additional diagnostic, not independent evidence or a tuning window.

### Error patterns

- ridge: monthly signed bias ranges from -422.4 to -183.8 SGD. Among towns with at least 200 final-block records, jurong west has the highest RMSE (635.0 SGD; 1,802 rows).
- catboost: monthly signed bias ranges from 9.4 to 40.3 SGD. Among towns with at least 200 final-block records, jurong west has the highest RMSE (557.4 SGD; 1,802 rows).
- mlp: monthly signed bias ranges from -17.7 to 116.3 SGD. Among towns with at least 200 final-block records, jurong west has the highest RMSE (577.7 SGD; 1,802 rows).

Ridge overpredicts the final-block rent by 337.8 SGD on average. The selected model underpredicts by 24.7 SGD on average. This calibration difference is worth discussing alongside RMSE. It does not identify a causal reason for rental changes.

The MLP needs only one or two selected epochs in this small experiment and varies across seeds. This suggests investigating optimization and temporal calibration before interpreting it as a definitive comparison of neural networks against trees.

The fixed-seed MLP has final-block RMSE 469.729 SGD despite worse earlier-fold selection scores than Ridge. Model rankings can change across periods; this is why per-fold results and forecasting behavior are retained alongside the aggregate selection score.

Positive signed bias means underprediction. Per-month, town, flat-type, and property-frequency metrics include sample counts and are saved as CSVs. The 30 largest errors per model are saved for case-level inspection; feature effects are associational, not causal.

## Limitations and deferred experiments

- School/mall features use supplied snapshots without historical opening dates. Historical amenity availability is uncertain; any gains do not establish a causal amenity effect.
- MRT remains the conservative pre-2021 proxy. Month-specific MRT features are deferred until reliable opening histories can be assembled and validated.
- COE/stocks are deferred because future observations cannot be assumed available at the prediction cutoff merely by lagging them one month.
- Historical rental summaries can become stale over the future test horizon. They do not use validation/test rents or recursively substitute predictions as actual rents.
- Tree models can struggle to extrapolate rental trends. A lower short-window score should be considered alongside monthly bias and the long-horizon diagnostic. The MLP search is deliberately small and its result does not establish that neural networks cannot perform well on this problem.
- The policy tie tolerances favor retaining potentially legitimate transactions and masking invalid source data. They are practical choices, not statistical significance thresholds.
- Selection involves multiple comparisons on two earlier folds; no confidence intervals or claims of statistically significant improvement are made.
- Ensembles, additional boosting libraries, and Ridge/tree residual models are deferred to keep the first experimental round focused. Their value would require a new validation comparison.

## Final feature artifacts and environment

The selected unencoded features are saved separately under `data/processed/improved/`, with a JSON contract. These full-training artifacts are for final prediction, not cross-validation: validation must rebuild repair maps and rental histories within its training window. Remaining numeric gaps are handled by the selected model pipeline.

Executed environment: Python 3.11.5, numpy 1.26.4, pandas 2.1.4, scikit-learn 1.2.2, CatBoost 1.2.10. The environment file defines supported dependencies; exact executed versions and source SHA-256 hashes are recorded in `experiments/results/final_refit.json`.

## Method references

- [CatBoost fit API](https://catboost.ai/docs/en/concepts/python-reference_catboostregressor_fit)
- [scikit-learn MLPRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)
