"""Chronological rental experiments shared by notebooks 06–10.

Original notebooks and processed files are read-only inputs. All experiment
outputs go to experiments/results; new submissions have distinct filenames.
"""
from pathlib import Path
import hashlib
import json
import platform
import re
import time
import unicodedata

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/results'
TARGET = 'MONTHLY_RENT'
KEYS = ['TOWN', 'BLOCK', 'STREET']
META = json.loads((ROOT / 'data/processed/hdb_mrt/preprocessing_metadata.json').read_text())
FEATURES = META['model_features']
CATS = META['categorical_features']
SCHOOLS = ['DIST_TO_NEAREST_SCHOOL_KM', 'SCHOOL_COUNT_WITHIN_500M',
           'SCHOOL_COUNT_WITHIN_1KM', 'DIST_TO_NEAREST_PRIMARY_SCHOOL_KM',
           'PRIMARY_SCHOOL_COUNT_WITHIN_500M', 'PRIMARY_SCHOOL_COUNT_WITHIN_1KM']
MALLS = ['DIST_TO_NEAREST_MALL_KM', 'MALL_COUNT_WITHIN_500M', 'MALL_COUNT_WITHIN_1KM']
HISTORY = ['HIST_TOWN_TYPE_MEAN', 'HIST_TOWN_TYPE_COUNT',
           'HIST_PROPERTY_MEAN', 'HIST_PROPERTY_COUNT']
FOLDS = [('2023_H2', '2023-07-01', '2024-01-01'),
         ('2024_H1', '2024-01-01', '2024-07-01')]
HOLDOUT = ('2024_H2_2025_Q1', '2024-07-01', '2025-04-01')


def normalize(value):
    if pd.isna(value):
        return np.nan
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(value)).strip().lower())


def prepare_core(raw):
    """Same deterministic features as notebook 03, without removing observations."""
    x = raw.copy()
    x.insert(0, 'ROW_ID', np.arange(len(x)))
    text = x.pop('RENT_APPROVAL_DATE').astype(str).str.strip()
    dates = pd.to_datetime(text, format='%Y-%m', errors='coerce')
    dates = dates.fillna(pd.to_datetime(text, format='%b-%y', errors='coerce'))
    if dates.isna().any():
        raise ValueError('Unrecognized rental dates')
    x['RENT_DATE'] = dates.dt.to_period('M').dt.to_timestamp()
    x['RENT_YEAR'], x['RENT_MONTH'] = dates.dt.year, dates.dt.month
    x['RENT_QUARTER'] = dates.dt.quarter
    x['MONTH_INDEX'] = (dates.dt.year - 2021) * 12 + dates.dt.month - 1
    x['MONTH_SIN'] = np.sin(2 * np.pi * x.RENT_MONTH / 12)
    x['MONTH_COS'] = np.cos(2 * np.pi * x.RENT_MONTH / 12)
    for c in ['TOWN', 'STREET', 'FLAT_MODEL', 'BLOCK', 'FLAT_TYPE']:
        x[c] = x[c].map(normalize)
    x['BLOCK'] = x.BLOCK.str.replace(r'\s+', '', regex=True)
    x['FLAT_TYPE'] = x.FLAT_TYPE.str.replace('-', ' ', regex=False).str.replace(r'\s+', ' ', regex=True).str.strip().str.replace(' ', '-', regex=False)
    x['PROPERTY_KEY'] = x[KEYS].fillna('').agg('|'.join, axis=1)
    x['TOWN_STREET'] = x[['TOWN', 'STREET']].fillna('').agg('|'.join, axis=1)
    x['FLAT_TYPE_ORDER'] = x.FLAT_TYPE.map({'1-room': 1, '2-room': 2, '3-room': 3, '4-room': 4, '5-room': 5, 'executive': 6})
    age = x.RENT_YEAR - pd.to_numeric(x.LEASE_COMMENCE_DATE, errors='coerce')
    invalid = age.lt(0) | age.gt(100)
    x['LEASE_AGE_INVALID'] = invalid.astype('int8')
    x['LEASE_AGE_AT_RENT'] = age.mask(invalid)
    x['LOG_FLOOR_AREA_SQM'] = np.log1p(pd.to_numeric(x.FLOOR_AREA_SQM))
    return x.drop(columns=['FURNISHED', 'FEE'])


def load_data():
    frames = []
    spatial = pd.read_csv(ROOT / 'data/processed/auxiliary/hdb_spatial_features.csv')
    assert not spatial.duplicated(KEYS).any()
    for split in ['train', 'test']:
        raw = pd.read_csv(ROOT / f'data/{split}.csv')
        core = prepare_core(raw)
        aux = pd.read_csv(ROOT / f'data/processed/auxiliary/{split}_hdb_mrt_features.csv', parse_dates=['RENT_DATE'])
        x = core.merge(aux, on=['ROW_ID', 'RENT_DATE'], validate='one_to_one', how='left', indicator=True)
        assert x.pop('_merge').eq('both').all()
        x = x.merge(spatial[KEYS + SCHOOLS + MALLS], on=KEYS, how='left', validate='many_to_one', indicator=True)
        assert x.pop('_merge').eq('both').all()
        for c in CATS:
            x[c] = x[c].fillna('__missing__').astype(str)
        assert x.ROW_ID.is_unique and len(x) == len(raw)
        assert not np.isinf(x[FEATURES + SCHOOLS + MALLS].select_dtypes('number')).any().any()
        frames.append(x)
    assert TARGET not in frames[1]
    return frames


def training_policy(frame, policy):
    """Deduplicate before fitting, never on validation/test rows."""
    cols = FEATURES + ['RENT_DATE']
    if policy == 'keep_all':
        return frame.copy()
    if policy == 'exact_only':
        return frame.drop_duplicates(cols + [TARGET]).copy()
    if policy == 'collapse_mean':
        # Preserve optional amenity columns from the representative row.
        ids = frame.groupby(cols, dropna=False, sort=False)[TARGET].agg(['mean']).reset_index()
        representatives = frame.drop_duplicates(cols).drop(columns=TARGET)
        return representatives.merge(ids.rename(columns={'mean': TARGET}), on=cols, validate='one_to_one')
    raise ValueError(policy)


class BlockRecovery:
    """Recover shared source attributes only from agreeing training addresses."""
    sources = {'LEASE_COMMENCE_DATE': 'LEASE_AGE_INVALID',
               'HDB_MAX_FLOOR': 'HDB_MAX_FLOOR_INVALID',
               'HDB_YEAR_COMPLETED': 'HDB_YEAR_COMPLETED_INVALID'}

    @staticmethod
    def mask(frame, column, flag):
        value = pd.to_numeric(frame[column], errors='coerce')
        bad = frame[flag].eq(1) | value.le(0)
        if column != 'HDB_MAX_FLOOR':
            age = frame.RENT_YEAR - value
            bad |= age.lt(0) | age.gt(100)
        return value.mask(bad)

    def fit(self, frame):
        self.maps = {}
        for c, flag in self.sources.items():
            temp = frame[KEYS].copy()
            temp[c] = self.mask(frame, c, flag)
            groups = temp.groupby(KEYS, dropna=False)[c]
            values = groups.first()
            self.maps[c] = values[groups.nunique().eq(1)]
        return self

    def transform(self, frame):
        x = frame.copy()
        index = pd.MultiIndex.from_frame(x[KEYS])
        for c, flag in self.sources.items():
            values = self.mask(x, c, flag)
            recovered = pd.Series(self.maps[c].reindex(index).to_numpy(), index=x.index)
            if c != 'HDB_MAX_FLOOR':
                age = x.RENT_YEAR - recovered
                recovered = recovered.mask(age.lt(0) | age.gt(100))
            x[c] = values.fillna(recovered)
        for source, derived in [('LEASE_COMMENCE_DATE', 'LEASE_AGE_AT_RENT'),
                                ('HDB_YEAR_COMPLETED', 'HDB_BUILDING_AGE_AT_RENT')]:
            age = x.RENT_YEAR - x[source]
            x[derived] = age.mask(age.lt(0) | age.gt(100))
        # Original invalid flags remain intact even when a value is recovered.
        return x


def history_features(train, other):
    """Train rows use earlier months; other rows use fixed training-cutoff history."""
    assert train.RENT_DATE.max() <= other.RENT_DATE.min()
    train, other = train.copy(), other.copy()
    specs = [(['TOWN', 'FLAT_TYPE'], 'TOWN_TYPE'), (['PROPERTY_KEY'], 'PROPERTY')]
    for keys, name in specs:
        mean_col, count_col = f'HIST_{name}_MEAN', f'HIST_{name}_COUNT'
        train[mean_col], train[count_col] = np.nan, 0.0
        # Compute summaries once, then use cumulative sums minus current month.
        monthly = train.groupby(keys + ['RENT_DATE'], dropna=False)[TARGET].agg(['sum', 'count']).reset_index().sort_values('RENT_DATE')
        groups = monthly.groupby(keys, dropna=False, sort=False)
        monthly['past_sum'] = groups['sum'].cumsum() - monthly['sum']
        monthly['past_count'] = groups['count'].cumsum() - monthly['count']
        monthly[mean_col] = monthly.past_sum / monthly.past_count.replace(0, np.nan)
        monthly[count_col] = monthly.past_count
        train = train.drop(columns=[mean_col, count_col]).merge(monthly[keys + ['RENT_DATE', mean_col, count_col]], on=keys + ['RENT_DATE'], validate='many_to_one', how='left')
        stats = train.groupby(keys, dropna=False)[TARGET].agg(['mean', 'count']).reset_index().rename(columns={'mean': mean_col, 'count': count_col})
        other = other.merge(stats, on=keys, how='left', validate='many_to_one')
        other[count_col] = other[count_col].fillna(0)
    train = train.sort_values('ROW_ID').reset_index(drop=True)
    other = other.sort_values('ROW_ID').reset_index(drop=True)
    return train, other


def features_for(group):
    return FEATURES + {'base': [], 'schools': SCHOOLS, 'malls': MALLS,
                       'schools_malls': SCHOOLS + MALLS, 'history': HISTORY}[group]


def prep_pair(train, valid, policy, imputation, group):
    train = training_policy(train, policy).sort_values(['RENT_DATE', 'ROW_ID'])
    valid = valid.sort_values('ROW_ID').copy()
    if imputation == 'block_recovery':
        repair = BlockRecovery().fit(train)
        train, valid = repair.transform(train), repair.transform(valid)
    if group == 'history':
        train, valid = history_features(train, valid)
        train = train.sort_values(['RENT_DATE', 'ROW_ID'])
    return train, valid


def encoder(features):
    cats = [c for c in CATS if c in features]
    nums = [c for c in features if c not in cats]
    return ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)), ('scale', StandardScaler())]), nums),
        ('categorical', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('encode', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=10, sparse_output=True))]), cats),
    ], sparse_threshold=1.0)


def score(y, prediction):
    return {'RMSE': float(np.sqrt(mean_squared_error(y, prediction))),
            'MAE': float(mean_absolute_error(y, prediction)),
            'bias_actual_minus_predicted': float(np.mean(np.asarray(y) - prediction))}


def internal_split(train):
    cutoff = train.RENT_DATE.max() - pd.DateOffset(months=2)
    a, b = train[train.RENT_DATE < cutoff], train[train.RENT_DATE >= cutoff]
    assert len(a) and len(b) and a.RENT_DATE.max() < b.RENT_DATE.min()
    return a, b


def fit_predict(model_name, train_raw, valid_raw, policy, imputation, group, params):
    """All adaptive iteration selection takes place inside outer training data."""
    features = features_for(group)
    start = time.perf_counter()
    info = {}
    if model_name == 'ridge':
        train, valid = prep_pair(train_raw, valid_raw, policy, imputation, group)
        model = Pipeline([('preprocess', encoder(features)), ('ridge', Ridge(alpha=params.get('alpha', 10), solver='lsqr', max_iter=10000, tol=1e-4))])
        model.fit(train[features], train[TARGET])
        predict = lambda x: model.predict(x[features])
    elif model_name == 'catboost':
        from catboost import CatBoostRegressor
        a, b = internal_split(train_raw)
        a, b = prep_pair(a, b, policy, imputation, group)
        def cat_frame(x):
            return x[features].copy()
        common = dict(loss_function='RMSE', random_seed=42, thread_count=4, has_time=True,
                      allow_writing_files=False, verbose=False, **params)
        probe = CatBoostRegressor(iterations=500, **common)
        probe.fit(cat_frame(a), a[TARGET], cat_features=CATS,
                  eval_set=(cat_frame(b), b[TARGET]), early_stopping_rounds=40)
        iterations = max(1, probe.get_best_iteration() + 1)
        train, valid = prep_pair(train_raw, valid_raw, policy, imputation, group)
        model = CatBoostRegressor(iterations=iterations, **common)
        model.fit(cat_frame(train), train[TARGET], cat_features=CATS)
        predict = lambda x: model.predict(cat_frame(x))
        info['selected_iterations'] = iterations
    elif model_name == 'mlp':
        a, b = internal_split(train_raw)
        a, b = prep_pair(a, b, policy, imputation, group)
        transform = encoder(features)
        xa, xb = transform.fit_transform(a[features]), transform.transform(b[features])
        scaler = StandardScaler().fit(a[[TARGET]])
        ya = scaler.transform(a[[TARGET]]).ravel()
        common = dict(hidden_layer_sizes=(64, 32), alpha=0.01, learning_rate_init=0.001,
                      batch_size=1024, early_stopping=False, **params)
        probe = MLPRegressor(**common)
        best, stale, best_epoch = np.inf, 0, 1
        curve = []
        for epoch in range(1, 41):
            probe.partial_fit(xa, ya)
            pred = scaler.inverse_transform(probe.predict(xb).reshape(-1, 1)).ravel()
            rmse = score(b[TARGET], pred)['RMSE']
            curve.append(rmse)
            if rmse < best - 0.1:
                best, stale, best_epoch = rmse, 0, epoch
            else:
                stale += 1
            if stale >= 6:
                break
        train, valid = prep_pair(train_raw, valid_raw, policy, imputation, group)
        transform = encoder(features)
        xt = transform.fit_transform(train[features])
        scaler = StandardScaler().fit(train[[TARGET]])
        yt = scaler.transform(train[[TARGET]]).ravel()
        model = MLPRegressor(**common)
        for _ in range(best_epoch):
            model.partial_fit(xt, yt)
        predict = lambda x: scaler.inverse_transform(model.predict(transform.transform(x[features])).reshape(-1, 1)).ravel()
        info.update(selected_iterations=best_epoch, internal_rmse_curve=curve)
    else:
        raise ValueError(model_name)
    predictions = predict(valid)
    assert np.isfinite(predictions).all()
    info.update(train_rows=len(train), evaluation_rows=len(valid), seconds=time.perf_counter() - start)
    details = valid[['ROW_ID', 'RENT_DATE', 'TOWN', 'FLAT_TYPE', 'PROPERTY_KEY']].copy()
    if TARGET in valid:
        details[TARGET] = valid[TARGET].to_numpy()
    details['prediction'] = predictions
    details['property_training_count'] = valid.PROPERTY_KEY.map(train.PROPERTY_KEY.value_counts()).fillna(0).to_numpy()
    return details, info


def save_json(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, default=str))


def evaluate(train, model, policy, imputation, group, params, folds=FOLDS):
    rows = []
    for name, start, end in folds:
        a = train[train.RENT_DATE < start]
        b = train[(train.RENT_DATE >= start) & (train.RENT_DATE < end)]
        assert a.RENT_DATE.max() < b.RENT_DATE.min()
        with threadpool_limits(limits=4):
            details, info = fit_predict(model, a, b, policy, imputation, group, params)
        rows.append(dict(model=model, policy=policy, imputation=imputation, group=group,
                         params=json.dumps(params, sort_keys=True), fold=name,
                         **score(details[TARGET], details.prediction), **info))
        print(f'{model} {policy} {imputation} {group} {params} {name}: RMSE={rows[-1]["RMSE"]:.3f}; {info["seconds"]:.1f}s', flush=True)
    return rows


def audit():
    train, test = load_data()
    raw = pd.read_csv(ROOT / 'data/train.csv')
    normalized = train[FEATURES + ['RENT_DATE', TARGET]]
    feature_groups = train.groupby(FEATURES + ['RENT_DATE'], dropna=False)[TARGET].agg(['size', 'nunique', 'min', 'max'])
    summary = dict(raw_rows=len(raw), test_rows=len(test), raw_missing_cells=int(raw.isna().sum().sum()),
                   raw_exact_matches=int(raw.duplicated().sum()),
                   normalized_exact_matches=int(normalized.duplicated().sum()),
                   feature_matches=int(train.duplicated(FEATURES + ['RENT_DATE']).sum()),
                   largest_feature_group=int(feature_groups['size'].max()),
                   groups_with_different_rents=int(feature_groups['nunique'].gt(1).sum()))
    save_json('data_audit.json', summary)
    invalid_rows = []
    for split, x in [('train', train), ('test', test)]:
        for source, flag in BlockRecovery.sources.items():
            invalid = BlockRecovery.mask(x, source, flag).isna()
            invalid_rows.append(dict(split=split, source=source, invalid_rows=int(invalid.sum()), affected_addresses=len(x.loc[invalid, KEYS].drop_duplicates())))
        for c in ['LEASE_AGE_AT_RENT', 'HDB_BUILDING_AGE_AT_RENT']:
            invalid_rows.append(dict(split=split, source=c, invalid_rows=int(x[c].isna().sum()), affected_addresses=len(x.loc[x[c].isna(), KEYS].drop_duplicates())))
    pd.DataFrame(invalid_rows).to_csv(OUT / 'invalid_values.csv', index=False)
    train.loc[train.LEASE_AGE_INVALID.eq(1), ['ROW_ID'] + KEYS + ['RENT_DATE', 'LEASE_COMMENCE_DATE', 'LEASE_AGE_AT_RENT']].head(30).to_csv(OUT / 'invalid_lease_examples.csv', index=False)
    recovery = []
    for name, start, end in FOLDS:
        a, b = train[train.RENT_DATE < start], train[(train.RENT_DATE >= start) & (train.RENT_DATE < end)]
        repair = BlockRecovery().fit(a)
        fixed = repair.transform(b)
        for c, flag in repair.sources.items():
            before = repair.mask(b, c, flag)
            recovery.append(dict(fold=name, source=c, invalid_before=int(before.isna().sum()), recovered=int((before.isna() & fixed[c].notna()).sum()), remaining_missing=int(fixed[c].isna().sum()), agreeing_training_addresses=len(repair.maps[c])))
    pd.DataFrame(recovery).to_csv(OUT / 'block_recovery_audit.csv', index=False)
    rows = []
    for policy in ['keep_all', 'exact_only', 'collapse_mean']:
        for imputation in ['median', 'block_recovery']:
            rows += evaluate(train, 'ridge', policy, imputation, 'base', {'alpha': 10})
            pd.DataFrame(rows).to_csv(OUT / 'preprocessing_scores.csv', index=False)
    results = pd.DataFrame(rows)
    means = results.groupby(['policy', 'imputation']).RMSE.mean()
    best_score = means.min()
    # Prefer retaining potentially real transactions when within 1 SGD of best.
    eligible = means[means <= best_score + 1].reset_index()
    eligible['rank'] = eligible.policy.map({'keep_all': 0, 'exact_only': 1, 'collapse_mean': 2})
    policy = eligible.sort_values(['rank', 'RMSE']).iloc[0].policy
    policy_scores = means.loc[policy]
    # Prefer masking invalid source years when its loss is negligible (<0.1 SGD).
    imputation = ('block_recovery' if policy_scores['block_recovery'] <= policy_scores.min() + 0.1
                  else policy_scores.idxmin())
    chosen = eligible[(eligible.policy == policy) & (eligible.imputation == imputation)].iloc[0]
    selection = dict(policy=chosen.policy, imputation=chosen.imputation,
                     selection_rule='Within 1 SGD of best early-fold mean RMSE, prefer keep_all then exact_only; prefer masking/recovery when within 0.1 SGD of best imputation.', mean_RMSE=float(chosen.RMSE))
    save_json('preprocessing_selection.json', selection)
    print('Selected preprocessing:', selection, flush=True)
    return results


def settings():
    return json.loads((OUT / 'preprocessing_selection.json').read_text())


def model_experiment(model, configs=None, groups=None):
    train, _ = load_data()
    choice = settings()
    policy, imputation = choice['policy'], choice['imputation']
    defaults = {'ridge': [{'alpha': a} for a in [10, 100]],
               'catboost': [dict(depth=6, learning_rate=0.08, l2_leaf_reg=3),
                            dict(depth=8, learning_rate=0.05, l2_leaf_reg=5)],
               'mlp': [dict(random_state=seed) for seed in [42, 123, 2026]]}
    configs = defaults[model] if configs is None else configs
    rows = []
    for params in configs:
        rows += evaluate(train, model, policy, imputation, 'base', params)
        pd.DataFrame(rows).to_csv(OUT / f'{model}_scores.csv', index=False)
    means = pd.DataFrame(rows).groupby('params').RMSE.mean()
    # MLP seed is fixed in advance; other seeds measure stability, not tuning.
    best = configs[0] if model == 'mlp' else json.loads(means.idxmin())
    if groups is None:
        groups = ['schools', 'malls', 'schools_malls', 'history'] if model != 'mlp' else []
    for group in groups:
        rows += evaluate(train, model, policy, imputation, group, best)
        pd.DataFrame(rows).to_csv(OUT / f'{model}_scores.csv', index=False)
    result = pd.DataFrame(rows)
    selection_scores = result[result.params.eq(json.dumps(best, sort_keys=True))].groupby('group').RMSE.mean()
    group = selection_scores.idxmin()
    # Keep base features if added features improve mean RMSE by less than 1 SGD.
    if selection_scores['base'] - selection_scores[group] < 1:
        group = 'base'
    save_json(f'{model}_selection.json', dict(model=model, params=best, group=group,
              policy=policy, imputation=imputation, mean_RMSE=float(selection_scores[group]),
              feature_selection_rule='Minimum earlier-fold mean RMSE; retain base if improvement <1 SGD.',
              seed_policy='MLP uses predeclared seed 42; other seeds are stability diagnostics.'))
    return result


def diagnostics(details, prefix):
    details = details.copy()
    details['residual'] = details[TARGET] - details.prediction
    details['absolute_error'] = details.residual.abs()
    details['squared_error'] = details.residual ** 2
    details['month'] = details.RENT_DATE.dt.strftime('%Y-%m')
    details['property_frequency'] = np.where(details.property_training_count >= 10, 'common_10plus', 'rare_under10')
    for c in ['month', 'TOWN', 'FLAT_TYPE', 'property_frequency']:
        table = details.groupby(c).agg(rows=(TARGET, 'size'), MAE=('absolute_error', 'mean'), MSE=('squared_error', 'mean'), bias_actual_minus_predicted=('residual', 'mean'))
        table['RMSE'] = np.sqrt(table.pop('MSE'))
        table.to_csv(OUT / f'{prefix}_errors_by_{c}.csv')
    details.nlargest(30, 'absolute_error').to_csv(OUT / f'{prefix}_largest_errors.csv', index=False)


def final_evaluation():
    train, test = load_data()
    selections = [json.loads((OUT / f'{model}_selection.json').read_text()) for model in ['ridge', 'catboost', 'mlp']]
    winner = min(selections, key=lambda s: s['mean_RMSE'])
    # Freeze choice using earlier folds before looking at final-block performance.
    save_json('final_selection.json', {**winner, 'selection_basis': 'earlier folds only; final block is diagnostic, not used to switch winners'})
    rows = []
    for s in selections:
        name, start, end = HOLDOUT
        a, b = train[train.RENT_DATE < start], train[(train.RENT_DATE >= start) & (train.RENT_DATE < end)]
        with threadpool_limits(limits=4):
            details, info = fit_predict(s['model'], a, b, s['policy'], s['imputation'], s['group'], s['params'])
        rows.append(dict(model=s['model'], evaluation='final_block', **score(details[TARGET], details.prediction), **info))
        details.to_csv(OUT / f'{s["model"]}_final_predictions.csv', index=False)
        diagnostics(details, s['model'])
        pd.DataFrame(rows).to_csv(OUT / 'final_scores.csv', index=False)
        print('Final block:', rows[-1], flush=True)
    for s in [selections[0], winner] if winner['model'] != 'ridge' else [winner]:
        # Earlier-origin 16-month diagnostic, after selection; overlaps other blocks.
        a = train[train.RENT_DATE < '2023-07-01']
        b = train[(train.RENT_DATE >= '2023-07-01') & (train.RENT_DATE < '2024-11-01')]
        with threadpool_limits(limits=4):
            details, info = fit_predict(s['model'], a, b, s['policy'], s['imputation'], s['group'], s['params'])
        rows.append(dict(model=s['model'], evaluation='16_month_backtest', **score(details[TARGET], details.prediction), **info))
        diagnostics(details, s['model'] + '_long_horizon')
    pd.DataFrame(rows).to_csv(OUT / 'final_scores.csv', index=False)
    with threadpool_limits(limits=4):
        predicted, info = fit_predict(winner['model'], train, test, winner['policy'], winner['imputation'], winner['group'], winner['params'])
    sample = pd.read_csv(ROOT / 'data/example-submission.csv')
    assert sample.columns.tolist() == ['Id', 'Predicted']
    assert predicted.ROW_ID.is_unique and set(sample.Id) == set(predicted.ROW_ID)
    submission = sample[['Id']].copy()
    submission['Predicted'] = submission.Id.map(predicted.set_index('ROW_ID').prediction)
    assert len(submission) == 50000 and np.isfinite(submission.Predicted).all()
    path = ROOT / f'submissions/{winner["model"]}_improved.csv'
    submission.to_csv(path, index=False)
    written = pd.read_csv(path)
    assert written.Id.equals(sample.Id) and np.isfinite(written.Predicted).all()
    prepared_train, prepared_test = prep_pair(train, test, winner['policy'], winner['imputation'], winner['group'])
    data_dir = ROOT / 'data/processed/improved'
    data_dir.mkdir(parents=True, exist_ok=True)
    columns = ['ROW_ID', 'RENT_DATE'] + features_for(winner['group'])
    prepared_train.sort_values('ROW_ID')[columns + [TARGET]].to_csv(data_dir / 'train_preprocessed.csv', index=False)
    prepared_test[columns].to_csv(data_dir / 'test_preprocessed.csv', index=False)
    contract = dict(model_features=features_for(winner['group']), target=TARGET,
                    categorical_features=CATS, non_feature_columns=['ROW_ID', 'RENT_DATE'],
                    train_rows=len(prepared_train), test_rows=len(prepared_test),
                    preprocessing=winner, remaining_missing='Model-fitted imputation/native handling; do not reuse full-data repair maps for validation.',
                    history_test_policy='Known training rents only, including March 2025 training records; no test targets used.')
    (data_dir / 'preprocessing_metadata.json').write_text(json.dumps(contract, indent=2))
    from catboost import __version__ as catboost_version
    source_paths = [ROOT / 'data/train.csv', ROOT / 'data/test.csv', Path(__file__)]
    fingerprints = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    save_json('final_refit.json', dict(submission=str(path.relative_to(ROOT)), **info,
              model_features=features_for(winner['group']), target=TARGET,
              categorical_features=CATS, python=platform.python_version(),
              numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__,
              catboost=catboost_version, sha256=fingerprints))
    return pd.DataFrame(rows)
