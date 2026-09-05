#!/usr/bin/env python3
"""Train the documented Ridge baseline and create a competition submission.

This script reproduces the final model in ``notebooks/05_ridge_baseline.ipynb``:

* feature columns loaded from the preprocessing metadata;
* fold-safe numeric imputation/scaling and sparse categorical one-hot encoding;
* Ridge regression with alpha=10, selected by expanding-window validation.

By default, inputs are read from ``data/processed`` and the submission is
written to ``submissions/ridge_baseline.csv``. Run ``python
scripts/create_ridge_submission.py --help`` for path and model overrides.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data/processed/hdb_mrt/train_preprocessed.csv"
DEFAULT_TEST_PATH = PROJECT_ROOT / "data/processed/hdb_mrt/test_preprocessed.csv"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "data/processed/hdb_mrt/preprocessing_metadata.json"
DEFAULT_SAMPLE_PATH = PROJECT_ROOT / "data/example-submission.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "submissions/ridge_baseline_hdb_mrt.csv"


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Train the Ridge baseline and create an Id,Predicted submission CSV."
    )
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--sample-submission", type=Path, default=DEFAULT_SAMPLE_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--alpha",
        type=float,
        default=10.0,
        help="Ridge L2 penalty. Default: 10, selected in the baseline notebook.",
    )
    parser.add_argument(
        "--min-category-frequency",
        type=int,
        default=10,
        help="Pool one-hot categories observed fewer than this many times. Default: 10.",
    )
    return parser.parse_args()


def get_feature_columns(metadata: dict) -> tuple[list[str], list[str]]:
    """Load the ordered feature contract from preprocessing metadata."""
    features = metadata.get("model_features")
    categorical_features = metadata.get("categorical_features")
    if not isinstance(features, list) or not features:
        raise ValueError("Metadata must contain a non-empty model_features list")
    if not isinstance(categorical_features, list):
        raise ValueError("Metadata must contain a categorical_features list")
    if len(features) != len(set(features)):
        raise ValueError("Metadata model_features contains duplicate columns")

    unknown_categorical = sorted(set(categorical_features) - set(features))
    if unknown_categorical:
        raise ValueError(
            f"Categorical columns are absent from model_features: {unknown_categorical}"
        )
    numeric_features = [
        feature for feature in features if feature not in set(categorical_features)
    ]
    return categorical_features.copy(), numeric_features


def build_pipeline(
    alpha: float,
    min_category_frequency: int,
    categorical_features: list[str],
    numeric_features: list[str],
) -> Pipeline:
    """Build the preprocessing and Ridge pipeline used by the notebook."""
    if alpha < 0:
        raise ValueError("--alpha must be non-negative")
    if min_category_frequency < 1:
        raise ValueError("--min-category-frequency must be at least 1")

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=min_category_frequency,
                    sparse_output=True,
                ),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "ridge",
                Ridge(alpha=alpha, solver="lsqr", max_iter=10_000, tol=1e-4),
            ),
        ]
    )


def validate_inputs(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame,
    target: str,
    row_id: str,
    features: list[str],
) -> None:
    """Fail early when processed data or the submission contract has changed."""
    required_train = set(features + [target, row_id])
    required_test = set(features + [row_id])
    missing_train = sorted(required_train - set(train.columns))
    missing_test = sorted(required_test - set(test.columns))
    if missing_train or missing_test:
        raise ValueError(
            f"Missing columns; train={missing_train or 'none'}, test={missing_test or 'none'}"
        )
    if target in test.columns:
        raise ValueError(f"Target column {target!r} must not be present in test data")
    if list(sample.columns) != ["Id", "Predicted"]:
        raise ValueError(
            "Sample submission must have exactly these ordered columns: Id, Predicted"
        )
    if not train[row_id].is_unique or not test[row_id].is_unique or not sample["Id"].is_unique:
        raise ValueError("Training, test, and sample-submission identifiers must be unique")
    if len(test) != len(sample):
        raise ValueError(
            f"Test/sample row-count mismatch: {len(test):,} versus {len(sample):,}"
        )
    if set(test[row_id]) != set(sample["Id"]):
        raise ValueError("Processed test ROW_ID values do not match sample-submission Id values")
    if not np.isfinite(train[target]).all():
        raise ValueError(f"Training target {target!r} contains missing or non-finite values")


def create_submission(args: argparse.Namespace) -> Path:
    """Fit the baseline, predict test rows, validate output, and write the CSV."""
    for path in (args.train, args.test, args.metadata, args.sample_submission):
        if not path.is_file():
            raise FileNotFoundError(f"Required input does not exist: {path}")

    metadata = json.loads(args.metadata.read_text())
    target = metadata["target"]
    row_id = metadata["row_id"]
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    sample = pd.read_csv(args.sample_submission)
    categorical_features, numeric_features = get_feature_columns(metadata)
    features = categorical_features + numeric_features
    validate_inputs(train, test, sample, target, row_id, features)

    print(
        f"Training Ridge(alpha={args.alpha:g}) on {len(train):,} rows with "
        f"{len(categorical_features)} categorical and {len(numeric_features)} numeric features..."
    )
    pipeline = build_pipeline(
        args.alpha,
        args.min_category_frequency,
        categorical_features,
        numeric_features,
    )
    pipeline.fit(train[features], train[target])
    predictions = pipeline.predict(test[features])

    prediction_by_id = pd.Series(predictions, index=test[row_id], name="Predicted")
    submission = sample[["Id"]].copy()
    submission["Predicted"] = submission["Id"].map(prediction_by_id)

    if submission["Predicted"].isna().any():
        raise ValueError("Submission contains IDs without predictions")
    if not np.isfinite(submission["Predicted"]).all():
        raise ValueError("Submission contains non-finite predictions")
    if list(submission.columns) != ["Id", "Predicted"] or len(submission) != len(sample):
        raise AssertionError("Submission schema or row count changed unexpectedly")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)

    # Verify the serialized artifact rather than only the in-memory frame.
    written = pd.read_csv(args.output)
    if list(written.columns) != ["Id", "Predicted"] or len(written) != len(sample):
        raise AssertionError("Written submission failed schema validation")
    if not written["Id"].equals(sample["Id"]):
        raise AssertionError("Written submission does not preserve sample-submission ID order")

    print(f"Saved {len(written):,} predictions to {args.output}")
    print(
        "Prediction summary: "
        f"min={written['Predicted'].min():,.2f}, "
        f"mean={written['Predicted'].mean():,.2f}, "
        f"max={written['Predicted'].max():,.2f}"
    )
    return args.output


def main() -> None:
    """CLI entry point."""
    create_submission(parse_args())


if __name__ == "__main__":
    main()
