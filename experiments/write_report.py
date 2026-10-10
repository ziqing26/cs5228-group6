"""Build the report from completed experiment outputs, without inventing results."""
import json
from pathlib import Path
import pandas as pd
from experiments.rental_experiments import ROOT, OUT


def write_report():
    audit = json.loads((OUT / 'data_audit.json').read_text())
    pre = json.loads((OUT / 'preprocessing_selection.json').read_text())
    winner = json.loads((OUT / 'final_selection.json').read_text())
    refit = json.loads((OUT / 'final_refit.json').read_text())
    scores = pd.read_csv(OUT / 'final_scores.csv')
    final = scores[scores.evaluation.eq('final_block')].set_index('model')
    ridge_rmse = float(final.loc['ridge', 'RMSE'])
    winner_rmse = float(final.loc[winner['model'], 'RMSE'])
    improvement = 100 * (ridge_rmse - winner_rmse) / ridge_rmse
    lines = ['# Rental baseline improvement report', '',
        f'The earlier-fold selection chose **{winner["model"]}** with '
        f'**{winner["group"]}** features. On the later diagnostic block its RMSE '
        f'is **{winner_rmse:.3f} SGD**, compared with **{ridge_rmse:.3f} SGD** '
        f'for the new Ridge reference ({improvement:.2f}% RMSE reduction). '
        'This comparison uses original individual rent targets and the same '
        'validation observations.', '',
        '## Scope and reproducibility', '',
        'This report documents the new experiments in notebooks 06–10. Notebooks '
        '01–05, original datasets, and baseline submissions were preserved. '
        'The question is whether preprocessing, nonlinear models, and additional '
        'amenity/history features improve future rental prediction without changing '
        'the evaluation targets.', '',
        'Run `python scripts/run_improvement_notebooks.py` from the repository root '
        'after installing `environment.yml`. Shared code is in '
        '`experiments/rental_experiments.py`. Machine-readable scores, selections, '
        'diagnostics, and source hashes are in `experiments/results/`.', '',
        '## Data and preprocessing audit', '',
        f'There are {audit["raw_rows"]:,} raw training observations and '
        f'{audit["test_rows"]:,} test observations. Raw training has '
        f'{audit["raw_missing_cells"]} missing cells. There are '
        f'{audit["raw_exact_matches"]:,} raw exact matches and '
        f'{audit["normalized_exact_matches"]:,} exact matches after deterministic '
        'normalization. Matching records cannot be confirmed as repeated transactions '
        'because actual flat identifiers are unavailable.', '',
        f'There are {audit["feature_matches"]:,} repeated feature rows; the largest '
        f'group has {audit["largest_feature_group"]} observations, and '
        f'{audit["groups_with_different_rents"]:,} feature groups contain different '
        'rents. Averaging these targets changes the training objective and weights. '
        'It is evaluated only on the training portion, never by smoothing validation targets.', '']
    preprocessing = pd.read_csv(OUT / 'preprocessing_scores.csv')
    for (policy, imp), val in preprocessing.groupby(['policy', 'imputation']).RMSE.mean().items():
        lines.append(f'- {policy}, {imp}: early-fold mean RMSE **{val:.3f} SGD**.')
    lines += ['', f'Selected policy: **{pre["policy"]}**, with **{pre["imputation"]}**. '
              + pre['selection_rule'] + ' These small imputation differences should not '
              'be interpreted as evidence of a substantial predictive improvement.', '',
        '### Invalid values and recovery', '',
        'A lease age below 0 or above 100 years is invalid. The legacy pipeline '
        'masked the age but left the source lease year exposed. The repair variant '
        'also masks the invalid source year. HDB maximum floors/completion years '
        'at or below zero are invalid; completion years producing negative or '
        'greater-than-100-year ages are also masked in the repair variant. '
        'Numeric parsing failures remain missing. These rules are heuristics, not '
        'proof of the correct underlying value.', '']
    for row in pd.read_csv(OUT / 'invalid_values.csv').itertuples():
        lines.append(f'- {row.split}, {row.source}: {row.invalid_rows:,} invalid/missing '
                     f'rows across {row.affected_addresses:,} addresses.')
    recovery = pd.read_csv(OUT / 'block_recovery_audit.csv')
    lines += ['', 'Recovery uses normalized town/block/street keys and requires all '
              'valid training values at that address to agree. Original invalid flags '
              'are retained. Derived ages are recomputed; remaining gaps use '
              'training-fitted medians for Ridge/MLP or native numeric missing-value '
              'handling for CatBoost.', '',
              f'Across the two earlier validation blocks, {int(recovery.recovered.sum())} '
              'source values were recovered. HDB attributes come from a shared block '
              'lookup, so an absent value for a block cannot usually be reconstructed '
              'from another transaction at that same block. Full audit details are in '
              '`experiments/results/block_recovery_audit.csv`.', '',
        '## Evaluation protocol', '',
        '- Selection windows: July–December 2023 and January–June 2024. Each model '
        'trains on strictly earlier months.',
        '- Final diagnostic block: July 2024–March 2025. It was already examined '
        'in the original baseline, so it is not completely untouched.',
        '- Every new experiment evaluates the same raw individual observations '
        'and original rent targets. Scores are not directly interchangeable with '
        'Notebook 05 scores, which use exact-deduplicated validation records.',
        '- Preprocessing, repair mappings, encodings, and target scaling are '
        'training-fitted. Tree counts/MLP epochs use the last three months inside '
        'each training portion, followed by refitting on the complete training portion.',
        '- MLP uses seed 42 for selection; seeds 123 and 2026 assess stability. '
        'No best-seed selection is performed.',
        '- Mean RMSE is an unweighted mean across the two selection blocks. '
        'Final and long-horizon metrics are pooled within each respective block.',
        '- The final model is frozen using earlier-fold results before final-block '
        'evaluation. Final scores do not trigger winner switching.', '',
        '## Model and feature experiments', '',
        'Ridge tests alpha 10 and 100. CatBoost tests a full 20-combination grid: '
        'depth [6, 8], learning rate [0.001, 0.005, 0.01, 0.05, 0.08], and '
        'L2 regularization [3, 5], including both previously tested configurations. '
        'Each combination is evaluated on both earlier chronological folds using base features; '
        'the lowest mean RMSE selects parameters before feature ablations. '
        'Its budget is 500 trees with 40-round internal early stopping. '
        'This fixed budget may disadvantage low learning rates that need more trees. '
        'CatBoost respects chronological input order and accepts categorical features '
        'directly. MLP uses 64 → 32 units, sparse one-hot inputs, standardized numeric '
        'inputs and rent targets, batch size 1024, regularization 0.01, and an '
        'internal selection budget of 40 epochs with six stale epochs.', '',
        'Ridge and CatBoost test schools, malls, both, and historical rental means/counts '
        'after choosing parameters on base features. History features use strictly '
        'earlier months for training rows and fixed training-cutoff summaries for '
        'evaluation rows. They summarize town × flat type and property, with '
        'training-only missing-value handling for unseen groups. MLP uses the original '
        '35 features for a controlled model comparison.', '']
    for model in ['ridge', 'catboost', 'mlp']:
        frame = pd.read_csv(OUT / f'{model}_scores.csv')
        selected = json.loads((OUT / f'{model}_selection.json').read_text())
        lines += ['', f'### {model}', '']
        for (group, params), value in frame.groupby(['group', 'params']).RMSE.mean().items():
            lines.append(f'- {group}, `{params}`: mean RMSE **{value:.3f} SGD**.')
        lines += ['', f'Selected feature group: **{selected["group"]}**. '
                  f'Selected parameters: `{json.dumps(selected["params"], sort_keys=True)}`. '
                  'Base features are retained if an added group improves mean RMSE '
                  'by less than 1 SGD.']
        if model == 'catboost':
            fixed = frame[frame.params.eq(json.dumps(selected['params'], sort_keys=True))]
            base_rmse = fixed[fixed.group.eq('base')].RMSE.mean()
            lines += ['', f'With the selected tree configuration, base-feature RMSE '
                      f'is {base_rmse:.3f} SGD and the selected feature-group RMSE '
                      f'is {selected["mean_RMSE"]:.3f} SGD. The extra feature gain is '
                      f'{base_rmse - selected["mean_RMSE"]:.3f} SGD; most of the '
                      'improvement over Ridge comes from the model change.']
    lines += ['', '## Final diagnostics and selection', '']
    for row in scores.itertuples():
        lines.append(f'- {row.model}, {row.evaluation}: RMSE **{row.RMSE:.3f} SGD**, '
                     f'MAE **{row.MAE:.3f} SGD**, actual-minus-predicted bias '
                     f'**{row.bias_actual_minus_predicted:.3f} SGD** '
                     f'({row.evaluation_rows:,} observations).')
    lines += ['', f'The earlier-fold winner is **{winner["model"]}** with '
              f'**{winner["group"]}** features. Its selected early-fold mean RMSE is '
              f'**{winner["mean_RMSE"]:.3f} SGD**. The separate submission is '
              f'`{refit["submission"]}` and contains 50,000 finite predictions in '
              'sample-submission ID order.', '',
              'The longer diagnostic predicts July 2023–October 2024 using data '
              'before July 2023. It tests a 16-month horizon and overlaps selection '
              'and final blocks; it is an additional diagnostic, not independent '
              'evidence or a tuning window.', '', '### Error patterns', '']
    for model in ['ridge', 'catboost', 'mlp']:
        month = pd.read_csv(OUT / f'{model}_errors_by_month.csv')
        town = pd.read_csv(OUT / f'{model}_errors_by_TOWN.csv')
        eligible = town[town.rows >= 200]
        worst = eligible.loc[eligible.RMSE.idxmax()]
        lines.append(f'- {model}: monthly signed bias ranges from '
                     f'{month.bias_actual_minus_predicted.min():.1f} to '
                     f'{month.bias_actual_minus_predicted.max():.1f} SGD. Among towns '
                     f'with at least 200 final-block records, {worst.TOWN} has the '
                     f'highest RMSE ({worst.RMSE:.1f} SGD; {int(worst.rows):,} rows).')
    ridge_bias = float(final.loc['ridge', 'bias_actual_minus_predicted'])
    winner_bias = float(final.loc[winner['model'], 'bias_actual_minus_predicted'])
    lines += ['', f'Ridge {"overpredicts" if ridge_bias < 0 else "underpredicts"} '
              f'the final-block rent by {abs(ridge_bias):.1f} SGD on average. '
              f'The selected model {"overpredicts" if winner_bias < 0 else "underpredicts"} '
              f'by {abs(winner_bias):.1f} SGD on average. This calibration difference '
              'is worth discussing alongside RMSE. It does not identify a causal '
              'reason for rental changes.', '',
              'The MLP needs only one or two selected epochs in this small experiment '
              'and varies across seeds. This suggests investigating optimization '
              'and temporal calibration before interpreting it as a definitive '
              'comparison of neural networks against trees.', '',
              f'The fixed-seed MLP has final-block RMSE '
              f'{float(final.loc["mlp", "RMSE"]):.3f} SGD despite worse earlier-fold '
              'selection scores than Ridge. Model rankings can change across '
              'periods; this is why per-fold results and forecasting behavior '
              'are retained alongside the aggregate selection score.']
    lines += ['', 'Positive signed bias means underprediction. Per-month, town, '
              'flat-type, and property-frequency metrics include sample counts and '
              'are saved as CSVs. The 30 largest errors per model are saved for '
              'case-level inspection; feature effects are associational, not causal.', '',
        '## Limitations and deferred experiments', '',
        '- School/mall features use supplied snapshots without historical opening '
        'dates. Historical amenity availability is uncertain; any gains do not '
        'establish a causal amenity effect.',
        '- MRT remains the conservative pre-2021 proxy. Month-specific MRT features '
        'are deferred until reliable opening histories can be assembled and validated.',
        '- COE/stocks are deferred because future observations cannot be assumed '
        'available at the prediction cutoff merely by lagging them one month.',
        '- Historical rental summaries can become stale over the future test '
        'horizon. They do not use validation/test rents or recursively substitute '
        'predictions as actual rents.',
        '- Tree models can struggle to extrapolate rental trends. A lower short-window '
        'score should be considered alongside monthly bias and the long-horizon '
        'diagnostic. The MLP search is deliberately small and its result does not '
        'establish that neural networks cannot perform well on this problem.',
        '- The policy tie tolerances favor retaining potentially legitimate '
        'transactions and masking invalid source data. They are practical choices, '
        'not statistical significance thresholds.',
        '- Selection involves multiple comparisons on two earlier folds; no '
        'confidence intervals or claims of statistically significant improvement '
        'are made.',
        '- Ensembles, additional boosting libraries, and Ridge/tree residual models '
        'are deferred to keep the first experimental round focused. Their value '
        'would require a new validation comparison.', '',
        '## Final feature artifacts and environment', '',
        'The selected unencoded features are saved separately under '
        '`data/processed/improved/`, with a JSON contract. These full-training '
        'artifacts are for final prediction, not cross-validation: validation must '
        'rebuild repair maps and rental histories within its training window. '
        'Remaining numeric gaps are handled by the selected model pipeline.', '',
        f'Executed environment: Python {refit["python"]}, numpy {refit["numpy"]}, '
        f'pandas {refit["pandas"]}, scikit-learn {refit["sklearn"]}, '
        f'CatBoost {refit["catboost"]}. The environment file defines supported '
        'dependencies; exact executed versions and source SHA-256 hashes are '
        'recorded in `experiments/results/final_refit.json`.', '',
        '## Method references', '',
        '- [CatBoost fit API](https://catboost.ai/docs/en/concepts/python-reference_catboostregressor_fit)',
        '- [scikit-learn MLPRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)', '']
    destination = ROOT / 'reports/improvement_report.md'
    destination.write_text('\n'.join(lines))
    return destination


if __name__ == '__main__':
    print(write_report())
