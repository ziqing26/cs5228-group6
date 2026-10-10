"""Checks for temporal leakage and preservation of observations."""
import unittest
import numpy as np
import pandas as pd
from experiments.rental_experiments import (BlockRecovery, FEATURES, TARGET,
                                           history_features, load_data, training_policy)


class RentalExperimentTests(unittest.TestCase):
    def test_history_excludes_same_month_and_future_targets(self):
        x = pd.DataFrame({'ROW_ID': [0, 1, 2, 3], 'RENT_DATE': pd.to_datetime(['2021-01-01', '2021-01-01', '2021-02-01', '2021-03-01']), 'TOWN': ['a'] * 4, 'FLAT_TYPE': ['3-room'] * 4, 'PROPERTY_KEY': ['a|1|road'] * 4, TARGET: [100, 300, 900, 700]})
        other = x.iloc[[3]].drop(columns=TARGET)
        a, b = history_features(x.iloc[:3], other)
        self.assertTrue(a.loc[:1, 'HIST_PROPERTY_MEAN'].isna().all())
        self.assertEqual(a.loc[2, 'HIST_PROPERTY_MEAN'], 200)
        self.assertAlmostEqual(b.loc[0, 'HIST_PROPERTY_MEAN'], 1300 / 3)
        self.assertEqual(b.loc[0, 'HIST_PROPERTY_COUNT'], 3)
        changed = x.iloc[:3].copy()
        changed.loc[2, TARGET] = 999999
        c, _ = history_features(changed, other)
        self.assertEqual(c.loc[2, 'HIST_PROPERTY_MEAN'], 200)

    def test_recovery_requires_agreement_and_ignores_evaluation_values(self):
        x = pd.DataFrame({'TOWN': ['a'] * 3, 'BLOCK': ['1', '1', '2'], 'STREET': ['road'] * 3, 'RENT_YEAR': [2024] * 3, 'LEASE_COMMENCE_DATE': [1980, 1981, 1990], 'LEASE_AGE_INVALID': [0] * 3, 'HDB_MAX_FLOOR': [10] * 3, 'HDB_MAX_FLOOR_INVALID': [0] * 3, 'HDB_YEAR_COMPLETED': [1980] * 3, 'HDB_YEAR_COMPLETED_INVALID': [0] * 3})
        valid = x.iloc[[0, 2]].copy()
        valid['LEASE_COMMENCE_DATE'] = 2099
        valid['LEASE_AGE_INVALID'] = 1
        result = BlockRecovery().fit(x).transform(valid)
        self.assertTrue(np.isnan(result.iloc[0].LEASE_COMMENCE_DATE))
        self.assertEqual(result.iloc[1].LEASE_COMMENCE_DATE, 1990)
        self.assertEqual(result.iloc[1].LEASE_AGE_AT_RENT, 34)
        self.assertTrue(result.LEASE_AGE_INVALID.eq(1).all())

    def test_reconstruction_matches_existing_core(self):
        from experiments.rental_experiments import ROOT
        train, test = load_data()
        core = pd.read_csv(ROOT / 'data/processed/core/train_preprocessed.csv', parse_dates=['RENT_DATE'])
        rebuilt = train.set_index('ROW_ID').loc[core.ROW_ID].reset_index()
        pd.testing.assert_frame_equal(rebuilt[core.columns], core, check_dtype=False, rtol=1e-6, atol=1e-6)
        self.assertEqual(len(train), 150000)
        self.assertEqual(len(test), 50000)
        self.assertEqual(len(training_policy(train, 'exact_only')), 149359)
        self.assertEqual(len(training_policy(training_policy(train, 'exact_only'), 'collapse_mean')), 128026)
        self.assertNotIn(TARGET, test)


if __name__ == '__main__':
    unittest.main()
