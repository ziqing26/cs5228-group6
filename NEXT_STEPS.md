# Next steps: improving the rental prediction baseline

## Objective

Improve the Ridge baseline through defensible preprocessing, focused model
comparisons, and feature experiments. Incorporate Chris's feedback as hypotheses
to investigate, rather than mandatory changes. Prioritize useful insights and a
clear report over extensive tuning for small RMSE improvements.

## Current reference

- Raw data: 150,000 training records and 50,000 test records.
- Core processed data: 149,359 training records after exact-record removal,
  with 19 model features and original `MONTHLY_RENT` targets.
- Saved HDB/MRT data: 128,026 training records with 35 features; identical
  feature vectors are collapsed and their rent targets averaged.
- Notebook `05_ridge_baseline.ipynb`: joins the core records with auxiliary
  HDB/MRT features, retaining 149,359 individual training observations and
  original targets. It compares core and enriched Ridge models fairly.
- All complete test variants contain 50,000 records without targets.

Keep the current Notebook 05 Ridge results as the reference. Do not assume its
duplicate policy is the final choice.

## 1. Audit duplicate handling

**Question:** Do identical observed records represent repeated entries or
different flats with the same recorded attributes?

Without actual flat numbers, this cannot be established from matching columns
alone. Describe them as identical observed records, not confirmed duplicate
transactions.

Compare these training policies:

1. Retain all original records.
2. Remove exact records only, documenting raw and normalization-induced matches
   separately.
3. Collapse identical feature vectors and average rents, as a separate experiment.

Apply each policy only within the training portion of each chronological fold.
Evaluate all policies on the same individual validation observations and original
rent targets. Do not collapse validation records or average their targets.
Report row counts, group sizes, and sensitivity of the results to the policy.

**Deliverable:** a justified duplicate policy and a concise explanation of its
uncertainty and effect on model performance.

## 2. Clarify invalid values and evaluate imputation

**Question:** Which values become missing during preprocessing, and can reliable
block-level information recover them?

- Document the exact rules for invalid lease ages and HDB attributes, including
  affected columns, counts, and examples.
- Distinguish missing values in raw data from missing values introduced by
  preprocessing. Check whether an invalid derived age also requires treatment of
  its source lease year.
- Inspect whether affected observations cluster within blocks and whether valid
  records within each block agree.
- Compare current training-fold median imputation with block-level recovery for
  attributes that should be shared and whose valid records agree. Identify blocks
  using normalized address keys, not block number alone.
- Use a training-fold median fallback when block recovery is unavailable or
  ambiguous. Recompute derived ages after recovering source attributes.
- Retain invalid/missing flags. Learn all recovery mappings and imputation
  statistics from the training fold only.

**Deliverable:** an invalid-value audit, an imputation comparison, and an explicit
description of the chosen policy in the report.

## 3. Establish the experiment protocol

- Use identical chronological evaluation observations across experiments.
- Preserve individual validation targets, regardless of training duplicate policy.
- Fit learned preprocessing, encodings, feature statistics, and imputation within
  each training fold.
- Use earlier chronological windows for model selection and reserve the latest
  block, July 2024–March 2025, for final comparison of new experiments. This block
  has already been examined in the baseline, so it is not a completely untouched
  holdout.
- Select early-stopping iterations using a chronological split inside the training
  portion, rather than the final evaluation block.
- Record per-fold RMSE and MAE, mean RMSE, training time, feature lists,
  preprocessing policy, parameters, and random seeds.
- Where data permits, add an earlier backtest with a horizon closer to the test
  horizon: test dates extend 16 months beyond the last training month.

**Deliverable:** a reusable evaluation setup and experiment log.

## 4. Compare models on a fixed feature set

After selecting the preprocessing policy, use the same 35 core + HDB/MRT features
to compare:

1. **Ridge:** retain the current reference pipeline.
2. **CatBoost:** the first nonlinear candidate, using categorical columns directly
   and a small search over depth, learning rate, and regularization.
3. **Small MLP:** scale numeric inputs, encode categorical inputs, use modest
   hidden layers such as 128 → 64, regularization, and several random seeds.

Use model-appropriate preprocessing while keeping observations and evaluation
splits consistent. None of these candidates is guaranteed to outperform Ridge on
future rents. Add LightGBM or XGBoost only if a second boosting implementation
answers a useful question or the initial results justify it.

**Deliverable:** a compact comparison explaining accuracy, stability, computational
cost, and model limitations.

## 5. Enhance features through ablation experiments

Add one feature group at a time to the reference and strongest candidate model.

### First priority: existing school and mall features

- Test school distances/counts, mall distances/counts, then both together.
- Join available features from `data/processed/auxiliary/hdb_spatial_features.csv`
  onto individual rental observations.
- Check whether source snapshots represent amenities available at the rental
  date; document any historical availability uncertainty.

### Second priority: temporal and rental-history features

- Replace the fixed pre-2021 MRT proxy with month-specific station availability
  if reliable opening dates are obtainable.
- Test historical rental summaries by town × flat type, street, or property,
  with observation counts and broader-group fallbacks.
- Construct training rental-history features from strictly earlier months.
  For validation/test, use only targets available before the prediction cutoff;
  never update histories using actual validation/test rents.
- Test town × time and flat type × time interactions for Ridge to allow different
  rental trends across segments.

### Lower priority: market features

- Test COE and stock features separately if time permits.
- Define availability across the future prediction horizon first. A one-month lag
  does not make future market observations available at the training cutoff.

Keep groups whose benefits are consistent and interpretable. Record unsuccessful
experiments when they explain why a group was excluded.

**Deliverable:** an ablation comparison and a documented final feature contract
shared by train and test.

## 6. Analyze errors and forecasting behavior

Go beyond aggregate RMSE:

- Examine RMSE, MAE, and signed bias by month, town, and flat type.
- Compare common and rare properties, with group sample counts.
- Inspect large errors for data issues, missing information, or unusual rents.
- Check whether gains hold across folds and longer forecasting horizons.
- Explain which locations or property types benefit from added features.

If models capture different patterns, test a simple Ridge/tree prediction blend.
If time extrapolation is a clear weakness, consider Ridge for the trend plus a
tree predicting its residuals, with residual training constructed inside each
fold without evaluation-target leakage.

**Deliverable:** findings about model behavior and limitations, not just a ranking
by score.

## 7. Write the report throughout

Start the data, EDA, preprocessing, and validation sections immediately. For every
experiment, record:

- The question or hypothesis.
- Data, features, preprocessing, model, and evaluation setup.
- Results and their consistency across periods.
- Interpretation and the resulting decision.

Explain duplicate uncertainty, invalid-value definitions, imputation scope, future
feature availability, and the limitations of the evaluation horizon. Finish with
a small, defensible set of comparisons instead of tuning extensively for marginal
gains.

## Milestones and completion criteria

1. **Preprocessing audit:** choose duplicate and imputation policies with evidence.
2. **Model comparison:** compare Ridge, CatBoost, and MLP on fixed features.
3. **Feature ablations:** prioritize schools/malls, then temporal/history features.
4. **Final evaluation:** assess shortlisted configurations and subgroup errors;
   select a model using accuracy, stability, and complexity together.
5. **Final artifacts:** refit on all training observations under the selected policy,
   generate 50,000 test predictions, verify submission IDs/order and finite values,
   and complete the report with reproducible settings.

Write the report alongside every milestone. Stop expanding the experiment scope
when the results support a defensible final choice and explain its practical
limitations.
