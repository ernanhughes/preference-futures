#!/usr/bin/env python3
"""
PreferenceFutures probe, v2.

This script tests the claim:

    Preferences can contain incremental information about future outcomes.

For a future target F, history/candidate features H,A, and preference Y:

    PFI = Loss(F | H, A) - Loss(F | H, A, Y)

Positive PFI means that adding the observed preference improved held-out
forecasting under the evaluated model, dataset, split, and loss.

Version 2 adds the controls required for the synthetic appendix:

1. Multiple random seeds.
2. A configurable synthetic preference-to-future effect.
3. A true null condition where preference has no future effect.
4. Paired hierarchical bootstrap confidence intervals:
   seeds are resampled, then complete held-out sessions are resampled within
   each selected seed.
5. Detailed and summary CSV outputs.

Tracks
------

synthetic
    Controlled session-continuation data. Use
    --synthetic-preference-effects 0,0.25,0.5,0.75
    to verify that measured PFI is near zero under the null and increases as
    preference information is injected.

arena
    Loads lmarena-ai/arena-human-preference-140k from Hugging Face, groups rows
    by evaluation_session_id, orders them by evaluation_order, and predicts
    whether another evaluation follows the current vote.

Examples
--------

Synthetic null and positive controls across ten seeds:

    python preference_futures_probe.py \
      --track synthetic \
      --seeds 1,2,3,4,5,6,7,8,9,10 \
      --synthetic-preference-effects 0,0.75 \
      --bootstrap-samples 2000 \
      --out synthetic_runs.csv \
      --summary-out synthetic_summary.csv

Synthetic effect calibration curve:

    python preference_futures_probe.py \
      --track synthetic \
      --seeds 1,2,3,4,5,6,7,8,9,10 \
      --synthetic-preference-effects 0,0.25,0.5,0.75 \
      --bootstrap-samples 2000 \
      --out synthetic_effect_runs.csv \
      --summary-out synthetic_effect_summary.csv

Arena:

    HF_TOKEN=... python preference_futures_probe.py \
      --track arena \
      --seeds 1,2,3,4,5 \
      --bootstrap-samples 2000 \
      --out arena_runs.csv \
      --summary-out arena_summary.csv

Dependencies
------------

    pip install pandas numpy scikit-learn datasets
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
import re
import sys
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data records
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ResultRow:
    track: str
    condition: str
    seed: int
    target: str
    feature_set: str
    n_train: int
    n_test: int
    n_train_groups: int
    n_test_groups: int
    loss_name: str
    loss: float
    brier: float | None = None
    auc: float | None = None
    accuracy: float | None = None
    synthetic_preference_effect: float | None = None
    synthetic_shared_latent_effect: float | None = None


@dataclasses.dataclass
class SummaryRow:
    track: str
    condition: str
    statistic: str
    n_seeds: int
    mean: float
    seed_std: float
    ci_low: float
    ci_high: float
    positive_seeds: int
    confidence_level: float
    bootstrap_samples: int
    synthetic_preference_effect: float | None = None
    synthetic_shared_latent_effect: float | None = None


@dataclasses.dataclass
class EvaluationBundle:
    rows: list[ResultRow]
    session_loss_stats: pd.DataFrame


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def sigmoid_scalar(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def parse_int_list(value: str | None, fallback: int) -> list[int]:
    if value is None or not value.strip():
        return [fallback]
    parsed = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not parsed:
        raise ValueError("--seeds did not contain any integers.")
    return parsed


def parse_float_list(value: str) -> list[float]:
    parsed = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not parsed:
        raise ValueError("The float list did not contain any values.")
    return parsed


def safe_jsonish_len(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, float) and math.isnan(value):
        return 0
    if isinstance(value, str):
        return len(value)
    try:
        return len(json.dumps(value, ensure_ascii=False))
    except Exception:
        return len(str(value))


def safe_token_count(value: Any) -> int:
    if value is None:
        return 0
    if not isinstance(value, str):
        try:
            value = json.dumps(value, ensure_ascii=False)
        except Exception:
            value = str(value)
    return len(re.findall(r"\w+", value))


def normalise_winner(value: Any) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip().lower()
    aliases = {
        "model_a": "a",
        "model a": "a",
        "winner_model_a": "a",
        "model_b": "b",
        "model b": "b",
        "winner_model_b": "b",
        "tie": "tie",
        "tie (bothbad)": "both_bad",
        "both_bad": "both_bad",
        "both bad": "both_bad",
    }
    if text in aliases:
        return aliases[text]
    if "both" in text and "bad" in text:
        return "both_bad"
    if text in {"a", "b"}:
        return text
    return text or "unknown"


def print_header(title: str) -> None:
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)


def log_loss_components(y_true: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    p = np.clip(probabilities.astype(float), 1e-12, 1.0 - 1e-12)
    y = y_true.astype(float)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def brier_components(y_true: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    return np.square(probabilities.astype(float) - y_true.astype(float))


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------


def fit_binary_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Train a probability-forecasting logistic baseline.

    Important:
    ``class_weight="balanced"`` is deliberately NOT used here. Balanced class
    weights change the effective class prior seen by the optimiser. That can be
    useful when the objective is minority-class recall, but the raw
    ``predict_proba`` values no longer estimate probabilities under the
    observed data distribution. PFI is evaluated with proper probabilistic
    scoring rules (log loss and Brier score), so the model must be trained
    against the real class prevalence.

    The returned diagnostics include a constant-prevalence null model. A useful
    forecasting model should beat that null on log loss and Brier score.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    X_train = train_df[feature_columns].copy()
    y_train = train_df[target_column].astype(int).to_numpy()
    X_test = test_df[feature_columns].copy()
    y_test = test_df[target_column].astype(int).to_numpy()

    if np.unique(y_train).size < 2:
        raise ValueError(
            f"Training split for target {target_column!r} contains only one class."
        )

    numeric_features = [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(X_train[column])
    ]
    categorical_features = [
        column for column in feature_columns if column not in numeric_features
    ]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_features:
        transformers.append(("num", numeric_pipe, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_pipe, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    # Do not rebalance classes when the quantity under evaluation is a
    # probability under the observed distribution.
    model = LogisticRegression(
        max_iter=2000,
        class_weight=None,
        random_state=seed,
    )

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipe.fit(X_train, y_train)
    probabilities = pipe.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    # Constant-probability baseline learned from the training split only.
    train_prevalence = float(np.mean(y_train))
    null_probabilities = np.full(
        shape=len(y_test),
        fill_value=np.clip(train_prevalence, 1e-12, 1.0 - 1e-12),
        dtype=float,
    )
    null_predictions = np.full(
        shape=len(y_test),
        fill_value=int(train_prevalence >= 0.5),
        dtype=int,
    )

    metrics: dict[str, float] = {
        "log_loss": float(log_loss(y_test, probabilities, labels=[0, 1])),
        "brier": float(brier_score_loss(y_test, probabilities)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, predictions)
        ),
        "average_precision": float(
            average_precision_score(y_test, probabilities)
        ),
        "train_prevalence": train_prevalence,
        "test_prevalence": float(np.mean(y_test)),
        "mean_predicted_probability": float(np.mean(probabilities)),
        "calibration_in_the_large": float(
            np.mean(probabilities) - np.mean(y_test)
        ),
        "null_log_loss": float(
            log_loss(y_test, null_probabilities, labels=[0, 1])
        ),
        "null_brier": float(
            brier_score_loss(y_test, null_probabilities)
        ),
        "null_accuracy": float(
            accuracy_score(y_test, null_predictions)
        ),
    }

    try:
        metrics["auc"] = float(roc_auc_score(y_test, probabilities))
    except ValueError:
        metrics["auc"] = float("nan")

    return probabilities, metrics


def shuffle_preference_columns(
    frame: pd.DataFrame,
    preference_features: list[str],
    seed: int,
) -> pd.DataFrame:
    """Shuffle the one-hot preference vector within evaluation-order buckets."""
    shuffled = frame.copy()
    if "evaluation_order" in shuffled.columns:
        pieces: list[pd.DataFrame] = []
        for bucket_index, (_, part) in enumerate(
            shuffled.groupby("evaluation_order", dropna=False)
        ):
            values = part[preference_features].sample(
                frac=1.0,
                random_state=seed + bucket_index,
            ).to_numpy()
            pieces.append(
                pd.DataFrame(
                    values,
                    columns=preference_features,
                    index=part.index,
                )
            )
        shuffled_values = pd.concat(pieces).sort_index()
        shuffled.loc[:, preference_features] = shuffled_values
    else:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(shuffled))
        shuffled.loc[:, preference_features] = (
            shuffled[preference_features].to_numpy()[indices]
        )
    return shuffled


def build_session_loss_stats(
    test_df: pd.DataFrame,
    *,
    group_column: str,
    target_column: str,
    seed: int,
    condition: str,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Build sufficient statistics for paired session-level bootstrapping."""
    y_true = test_df[target_column].astype(int).to_numpy()
    row_stats = pd.DataFrame(
        {
            "group_id": test_df[group_column].astype(str).to_numpy(),
            "n_rows": 1,
        },
        index=test_df.index,
    )

    for model_name, probability in predictions.items():
        row_stats[f"{model_name}_log_sum"] = log_loss_components(y_true, probability)
        row_stats[f"{model_name}_brier_sum"] = brier_components(y_true, probability)

    grouped = row_stats.groupby("group_id", as_index=False).sum(numeric_only=True)
    grouped["seed"] = seed
    grouped["condition"] = condition
    return grouped


def evaluate_binary_feature_sets(
    df: pd.DataFrame,
    *,
    group_column: str,
    target_column: str,
    base_features: list[str],
    preference_features: list[str],
    track: str,
    condition: str,
    seed: int,
    test_fraction: float = 0.2,
    synthetic_preference_effect: float | None = None,
    synthetic_shared_latent_effect: float | None = None,
) -> EvaluationBundle:
    """Evaluate no-preference, preference-only, full, and shuffled controls."""
    rng = np.random.default_rng(seed)

    groups = np.asarray(sorted(df[group_column].dropna().astype(str).unique()))
    if groups.size < 2:
        raise ValueError("At least two groups are required for a grouped split.")
    rng.shuffle(groups)

    n_test_groups = max(1, int(round(len(groups) * test_fraction)))
    n_test_groups = min(n_test_groups, len(groups) - 1)
    test_groups = set(groups[:n_test_groups])

    group_values = df[group_column].astype(str)
    train_df = df[~group_values.isin(test_groups)].copy()
    test_df = df[group_values.isin(test_groups)].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Empty train/test split. Check grouping and dataset size.")

    feature_sets = {
        "history_candidate_no_preference": base_features,
        "preference_only": preference_features,
        "history_candidate_plus_preference": base_features + preference_features,
    }

    rows: list[ResultRow] = []
    predictions: dict[str, np.ndarray] = {}

    for name, columns in feature_sets.items():
        probability, metrics = fit_binary_model(
            train_df,
            test_df,
            columns,
            target_column,
            seed,
        )
        predictions[name] = probability
        rows.append(
            ResultRow(
                track=track,
                condition=condition,
                seed=seed,
                target=target_column,
                feature_set=name,
                n_train=len(train_df),
                n_test=len(test_df),
                n_train_groups=train_df[group_column].astype(str).nunique(),
                n_test_groups=test_df[group_column].astype(str).nunique(),
                loss_name="log_loss",
                loss=metrics["log_loss"],
                brier=metrics["brier"],
                auc=metrics["auc"],
                accuracy=metrics["accuracy"],
                synthetic_preference_effect=synthetic_preference_effect,
                synthetic_shared_latent_effect=synthetic_shared_latent_effect,
            )
        )

    shuffled_train = shuffle_preference_columns(
        train_df, preference_features, seed=seed + 10_000
    )
    shuffled_test = shuffle_preference_columns(
        test_df, preference_features, seed=seed + 20_000
    )

    shuffled_probability, shuffled_metrics = fit_binary_model(
        shuffled_train,
        shuffled_test,
        base_features + preference_features,
        target_column,
        seed,
    )
    shuffled_name = "history_candidate_plus_shuffled_preference"
    predictions[shuffled_name] = shuffled_probability
    rows.append(
        ResultRow(
            track=track,
            condition=condition,
            seed=seed,
            target=target_column,
            feature_set=shuffled_name,
            n_train=len(shuffled_train),
            n_test=len(shuffled_test),
            n_train_groups=shuffled_train[group_column].astype(str).nunique(),
            n_test_groups=shuffled_test[group_column].astype(str).nunique(),
            loss_name="log_loss",
            loss=shuffled_metrics["log_loss"],
            brier=shuffled_metrics["brier"],
            auc=shuffled_metrics["auc"],
            accuracy=shuffled_metrics["accuracy"],
            synthetic_preference_effect=synthetic_preference_effect,
            synthetic_shared_latent_effect=synthetic_shared_latent_effect,
        )
    )

    stats = build_session_loss_stats(
        test_df,
        group_column=group_column,
        target_column=target_column,
        seed=seed,
        condition=condition,
        predictions={
            "no_pref": predictions["history_candidate_no_preference"],
            "full": predictions["history_candidate_plus_preference"],
            "shuffled": predictions[shuffled_name],
        },
    )

    return EvaluationBundle(rows=rows, session_loss_stats=stats)


# ---------------------------------------------------------------------------
# Bootstrap and aggregation
# ---------------------------------------------------------------------------


def statistic_from_session_stats(stats: pd.DataFrame, statistic: str) -> float:
    total_rows = float(stats["n_rows"].sum())
    if total_rows <= 0:
        return float("nan")

    if statistic == "pfi_log_loss":
        return float(
            (
                stats["no_pref_log_sum"].sum()
                - stats["full_log_sum"].sum()
            )
            / total_rows
        )
    if statistic == "pfi_brier":
        return float(
            (
                stats["no_pref_brier_sum"].sum()
                - stats["full_brier_sum"].sum()
            )
            / total_rows
        )
    if statistic == "shuffle_gap_log_loss":
        return float(
            (
                stats["shuffled_log_sum"].sum()
                - stats["full_log_sum"].sum()
            )
            / total_rows
        )
    if statistic == "shuffle_gap_brier":
        return float(
            (
                stats["shuffled_brier_sum"].sum()
                - stats["full_brier_sum"].sum()
            )
            / total_rows
        )
    raise ValueError(f"Unknown statistic: {statistic}")


def hierarchical_bootstrap_interval(
    stats: pd.DataFrame,
    *,
    statistic: str,
    samples: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float]:
    """Resample seeds, then complete held-out sessions within each seed.

    This implementation operates on NumPy sufficient-statistic arrays rather
    than repeatedly concatenating pandas frames, keeping publication-sized
    bootstrap runs practical.
    """
    if samples <= 0:
        return float("nan"), float("nan")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("--confidence-level must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    seed_values = np.asarray(sorted(stats["seed"].unique()), dtype=int)
    if seed_values.size == 0:
        return float("nan"), float("nan")

    columns = [
        "n_rows",
        "no_pref_log_sum",
        "full_log_sum",
        "shuffled_log_sum",
        "no_pref_brier_sum",
        "full_brier_sum",
        "shuffled_brier_sum",
    ]
    by_seed = {
        int(seed_value): stats.loc[
            stats["seed"] == seed_value, columns
        ].to_numpy(dtype=float)
        for seed_value in seed_values
    }

    def from_totals(totals: np.ndarray) -> float:
        n_rows = totals[0]
        if n_rows <= 0:
            return float("nan")
        if statistic == "pfi_log_loss":
            return float((totals[1] - totals[2]) / n_rows)
        if statistic == "shuffle_gap_log_loss":
            return float((totals[3] - totals[2]) / n_rows)
        if statistic == "pfi_brier":
            return float((totals[4] - totals[5]) / n_rows)
        if statistic == "shuffle_gap_brier":
            return float((totals[6] - totals[5]) / n_rows)
        raise ValueError(f"Unknown statistic: {statistic}")

    draws = np.empty(samples, dtype=float)
    for draw_index in range(samples):
        sampled_seed_values = rng.choice(
            seed_values, size=len(seed_values), replace=True
        )
        totals = np.zeros(len(columns), dtype=float)
        for sampled_seed in sampled_seed_values:
            seed_array = by_seed[int(sampled_seed)]
            row_indices = rng.integers(0, len(seed_array), size=len(seed_array))
            totals += seed_array[row_indices].sum(axis=0)
        draws[draw_index] = from_totals(totals)

    alpha = 1.0 - confidence_level
    low = float(np.quantile(draws, alpha / 2.0))
    high = float(np.quantile(draws, 1.0 - alpha / 2.0))
    return low, high


def build_summary_rows(
    detailed_rows: list[ResultRow],
    session_stats: pd.DataFrame,
    *,
    track: str,
    condition: str,
    bootstrap_samples: int,
    confidence_level: float,
    bootstrap_seed: int,
    synthetic_preference_effect: float | None,
    synthetic_shared_latent_effect: float | None,
) -> list[SummaryRow]:
    result_df = pd.DataFrame([dataclasses.asdict(row) for row in detailed_rows])
    seed_values = sorted(result_df["seed"].unique())

    per_seed: dict[str, list[float]] = {
        "pfi_log_loss": [],
        "pfi_brier": [],
        "shuffle_gap_log_loss": [],
        "shuffle_gap_brier": [],
    }

    for seed_value in seed_values:
        subset = result_df[result_df["seed"] == seed_value]
        indexed = subset.set_index("feature_set")
        no_pref = indexed.loc["history_candidate_no_preference"]
        full = indexed.loc["history_candidate_plus_preference"]
        shuffled = indexed.loc["history_candidate_plus_shuffled_preference"]

        per_seed["pfi_log_loss"].append(float(no_pref["loss"] - full["loss"]))
        per_seed["pfi_brier"].append(float(no_pref["brier"] - full["brier"]))
        per_seed["shuffle_gap_log_loss"].append(
            float(shuffled["loss"] - full["loss"])
        )
        per_seed["shuffle_gap_brier"].append(
            float(shuffled["brier"] - full["brier"])
        )

    rows: list[SummaryRow] = []
    for statistic, values_list in per_seed.items():
        values = np.asarray(values_list, dtype=float)
        low, high = hierarchical_bootstrap_interval(
            session_stats,
            statistic=statistic,
            samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=bootstrap_seed + sum(ord(ch) for ch in statistic),
        )
        rows.append(
            SummaryRow(
                track=track,
                condition=condition,
                statistic=statistic,
                n_seeds=len(values),
                mean=float(np.mean(values)),
                seed_std=float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                ci_low=low,
                ci_high=high,
                positive_seeds=int(np.sum(values > 0)),
                confidence_level=confidence_level,
                bootstrap_samples=bootstrap_samples,
                synthetic_preference_effect=synthetic_preference_effect,
                synthetic_shared_latent_effect=synthetic_shared_latent_effect,
            )
        )
    return rows


def print_condition_results(
    rows: list[ResultRow],
    summary_rows: list[SummaryRow],
) -> None:
    result_df = pd.DataFrame([dataclasses.asdict(row) for row in rows])
    metric_columns = ["loss", "brier", "auc", "accuracy"]
    aggregate = (
        result_df.groupby("feature_set")[metric_columns]
        .agg(["mean", "std"])
        .sort_index()
    )

    print("\nFeature-set metrics across seeds")
    print(aggregate.to_string(float_format=lambda value: f"{value:.6f}"))

    summary_df = pd.DataFrame(
        [dataclasses.asdict(row) for row in summary_rows]
    )
    display_columns = [
        "statistic",
        "n_seeds",
        "mean",
        "seed_std",
        "ci_low",
        "ci_high",
        "positive_seeds",
    ]
    print("\nPFI and shuffled-control summary")
    print(
        summary_df[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )


# ---------------------------------------------------------------------------
# Synthetic track
# ---------------------------------------------------------------------------


def make_synthetic_dataset(
    n_sessions: int,
    rows_per_session: int,
    seed: int,
    preference_effect: float,
    shared_latent_effect: float,
) -> pd.DataFrame:
    """Generate controlled future-bearing preference trajectories.

    Preference generation:
        Y depends on visible candidate features and a latent session regime.

    Future generation:
        F depends on visible context, independent patience, and
        preference_effect * Y.

    By default shared_latent_effect is zero. Therefore, when
    preference_effect is also zero, Y is conditionally independent of F given
    the visible base features: this is the strict null condition.

    Setting shared_latent_effect above zero creates an optional observational
    confounding experiment in which the preference reveals additional
    information about a latent regime that also affects the future.
    """
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []

    for session_id in range(n_sessions):
        latent_regime = float(rng.normal())
        patience = float(rng.normal())
        session_topic = str(
            rng.choice(["code", "writing", "reasoning", "qa"])
        )

        for order in range(rows_per_session):
            difficulty = float(rng.normal() + 0.25 * order)
            response_gap = float(rng.normal() + 0.45 * latent_regime)
            candidate_length_delta = float(rng.normal() + 0.2 * difficulty)
            history_length = max(
                1, int(100 + 20 * order + rng.normal(0, 15))
            )

            preference_logit = (
                0.8 * response_gap
                - 0.25 * candidate_length_delta
                + 0.65 * latent_regime
                + float(rng.normal(0, 0.75))
            )
            preference_a = int(
                rng.random() < sigmoid_scalar(preference_logit)
            )

            future_logit = (
                -0.25 * order
                + 0.45 * difficulty
                + 0.35 * patience
                + preference_effect * preference_a
                + shared_latent_effect * latent_regime
                + float(rng.normal(0, 0.8))
            )
            continues = int(rng.random() < sigmoid_scalar(future_logit))

            records.append(
                {
                    "evaluation_session_id": f"s{session_id}",
                    "evaluation_order": order,
                    "topic": session_topic,
                    "history_len": history_length,
                    "candidate_a_len": (
                        120 + 20 * response_gap + rng.normal(0, 10)
                    ),
                    "candidate_b_len": (
                        120 - 20 * response_gap + rng.normal(0, 10)
                    ),
                    "candidate_len_delta": candidate_length_delta,
                    "response_gap_proxy": response_gap,
                    "difficulty_proxy": difficulty,
                    "winner_norm": "a" if preference_a else "b",
                    "pref_a": preference_a,
                    "pref_b": 1 - preference_a,
                    "pref_tie": 0,
                    "pref_both_bad": 0,
                    "session_continues_after_vote": continues,
                }
            )

            if not continues:
                break

    return pd.DataFrame(records)


def synthetic_features() -> tuple[list[str], list[str]]:
    base_features = [
        "evaluation_order",
        "topic",
        "history_len",
        "candidate_a_len",
        "candidate_b_len",
        "candidate_len_delta",
        "response_gap_proxy",
        "difficulty_proxy",
    ]
    preference_features = ["pref_a", "pref_b", "pref_tie", "pref_both_bad"]
    return base_features, preference_features


def run_synthetic_matrix(
    args: argparse.Namespace,
    seeds: Sequence[int],
) -> tuple[list[ResultRow], list[SummaryRow]]:
    effects = parse_float_list(args.synthetic_preference_effects)
    base_features, preference_features = synthetic_features()

    all_rows: list[ResultRow] = []
    all_summaries: list[SummaryRow] = []

    for effect_index, effect in enumerate(effects):
        condition = (
            f"synthetic_effect_{effect:g}"
            f"_shared_latent_{args.synthetic_shared_latent_effect:g}"
        )
        condition_rows: list[ResultRow] = []
        condition_stats: list[pd.DataFrame] = []

        print_header(
            f"Synthetic condition: preference effect={effect:g}, "
            f"shared latent effect={args.synthetic_shared_latent_effect:g}"
        )

        for seed in seeds:
            df = make_synthetic_dataset(
                n_sessions=args.synthetic_sessions,
                rows_per_session=args.synthetic_max_rounds,
                seed=seed,
                preference_effect=effect,
                shared_latent_effect=args.synthetic_shared_latent_effect,
            )
            bundle = evaluate_binary_feature_sets(
                df,
                group_column="evaluation_session_id",
                target_column="session_continues_after_vote",
                base_features=base_features,
                preference_features=preference_features,
                track="synthetic",
                condition=condition,
                seed=seed,
                test_fraction=args.test_fraction,
                synthetic_preference_effect=effect,
                synthetic_shared_latent_effect=args.synthetic_shared_latent_effect,
            )
            condition_rows.extend(bundle.rows)
            condition_stats.append(bundle.session_loss_stats)

            target_rate = float(
                df["session_continues_after_vote"].mean()
            )
            print(
                f"seed={seed:>4} rows={len(df):>6,} "
                f"sessions={df['evaluation_session_id'].nunique():>5,} "
                f"continuation_rate={target_rate:.4f}"
            )

        combined_stats = pd.concat(condition_stats, ignore_index=True)
        summaries = build_summary_rows(
            condition_rows,
            combined_stats,
            track="synthetic",
            condition=condition,
            bootstrap_samples=args.bootstrap_samples,
            confidence_level=args.confidence_level,
            bootstrap_seed=args.bootstrap_seed + effect_index * 1000,
            synthetic_preference_effect=effect,
            synthetic_shared_latent_effect=args.synthetic_shared_latent_effect,
        )

        print_condition_results(condition_rows, summaries)
        all_rows.extend(condition_rows)
        all_summaries.extend(summaries)

    return all_rows, all_summaries


# ---------------------------------------------------------------------------
# Arena track
# ---------------------------------------------------------------------------


def load_arena_dataframe(
    limit_rows: int | None,
    *,
    sample_seed: int = 1729,
) -> pd.DataFrame:
    """Load Arena data, preserving complete evaluation sessions.

    The old smoke-test implementation selected the first ``limit_rows`` rows
    before constructing the continuation target. That can cut a session in
    half and falsely label its last retained row as a terminal event.

    This version loads the session-id column first, samples complete sessions,
    and only then materialises the selected rows.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install with `pip install datasets`."
        ) from exc

    repo_id = "lmarena-ai/arena-human-preference-140k"
    dataset = load_dataset(repo_id, split="train", token=True)

    if limit_rows is not None and limit_rows > 0 and limit_rows < len(dataset):
        session_ids = np.asarray(
            dataset["evaluation_session_id"],
            dtype=object,
        )

        unique_sessions, counts = np.unique(
            session_ids,
            return_counts=True,
        )
        rng = np.random.default_rng(sample_seed)
        order = rng.permutation(len(unique_sessions))

        selected_sessions: list[object] = []
        selected_rows = 0
        for index in order:
            session_size = int(counts[index])
            if selected_sessions and selected_rows + session_size > limit_rows:
                continue
            selected_sessions.append(unique_sessions[index])
            selected_rows += session_size
            if selected_rows >= limit_rows:
                break

        selected_set = set(selected_sessions)
        selected_indices = [
            index
            for index, session_id in enumerate(session_ids)
            if session_id in selected_set
        ]
        dataset = dataset.select(selected_indices)

    return dataset.to_pandas()


def pick_first_existing(
    columns: Iterable[str], candidates: list[str]
) -> str | None:
    available = set(columns)
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def prepare_arena_future_dataset(
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    required_candidates = {
        "evaluation_session_id": ["evaluation_session_id", "session_id"],
        "evaluation_order": ["evaluation_order", "order", "turn"],
        "winner": ["winner", "vote", "preference"],
    }

    resolved: dict[str, str] = {}
    for logical_name, candidate_names in required_candidates.items():
        found = pick_first_existing(raw.columns, candidate_names)
        if found is None:
            schema = "\n".join(f"  - {column}" for column in raw.columns)
            raise ValueError(
                f"Could not find required logical field {logical_name!r}. "
                f"Available columns:\n{schema}"
            )
        resolved[logical_name] = found

    df = raw.copy()
    df["_session_id"] = df[resolved["evaluation_session_id"]].astype(str)
    df["_order"] = pd.to_numeric(
        df[resolved["evaluation_order"]], errors="coerce"
    )
    df = df.dropna(subset=["_order"]).copy()
    df["_order"] = df["_order"].astype(int)
    df["winner_norm"] = df[resolved["winner"]].map(normalise_winner)

    df = df.sort_values(["_session_id", "_order"]).copy()
    group_sizes = df.groupby("_session_id")["_order"].transform("size")
    group_rank = df.groupby("_session_id").cumcount()
    df["session_continues_after_vote"] = (
        group_rank < group_sizes - 1
    ).astype(int)

    conversation_a = pick_first_existing(
        df.columns,
        ["conversation_a", "messages_a", "response_a", "answer_a"],
    )
    conversation_b = pick_first_existing(
        df.columns,
        ["conversation_b", "messages_b", "response_b", "answer_b"],
    )
    full_conversation = pick_first_existing(
        df.columns,
        ["full_conversation", "conversation", "messages", "prompt"],
    )
    model_a = pick_first_existing(
        df.columns, ["model_a", "model_a_name"]
    )
    model_b = pick_first_existing(
        df.columns, ["model_b", "model_b_name"]
    )
    category = pick_first_existing(
        df.columns,
        ["category", "categories", "turn_category", "language"],
    )

    df["candidate_a_chars"] = (
        df[conversation_a].map(safe_jsonish_len) if conversation_a else 0
    )
    df["candidate_a_tokens"] = (
        df[conversation_a].map(safe_token_count) if conversation_a else 0
    )
    df["candidate_b_chars"] = (
        df[conversation_b].map(safe_jsonish_len) if conversation_b else 0
    )
    df["candidate_b_tokens"] = (
        df[conversation_b].map(safe_token_count) if conversation_b else 0
    )

    if full_conversation:
        df["history_chars"] = df[full_conversation].map(safe_jsonish_len)
        df["history_tokens"] = df[full_conversation].map(safe_token_count)
    else:
        df["history_chars"] = (
            df["candidate_a_chars"] + df["candidate_b_chars"]
        )
        df["history_tokens"] = (
            df["candidate_a_tokens"] + df["candidate_b_tokens"]
        )

    df["candidate_char_delta"] = (
        df["candidate_a_chars"] - df["candidate_b_chars"]
    )
    df["candidate_token_delta"] = (
        df["candidate_a_tokens"] - df["candidate_b_tokens"]
    )

    df["model_a_feature"] = (
        df[model_a].astype(str) if model_a else "unknown"
    )
    df["model_b_feature"] = (
        df[model_b].astype(str) if model_b else "unknown"
    )
    df["category_feature"] = (
        df[category].astype(str) if category else "unknown"
    )

    for label in ["a", "b", "tie", "both_bad"]:
        df[f"pref_{label}"] = (
            df["winner_norm"] == label
        ).astype(int)

    # Canonicalise without renaming temporary columns onto existing source
    # columns. The Arena dataset already uses ``evaluation_session_id`` and
    # ``evaluation_order``; renaming ``_session_id``/``_order`` to those names
    # would create duplicate column labels. In pandas, selecting a duplicated
    # label returns a DataFrame rather than a Series, which breaks operations
    # such as ``value_counts()`` and ``groupby()``.
    prepared = df.copy()
    prepared["evaluation_session_id"] = prepared["_session_id"].astype(str)
    prepared["evaluation_order"] = prepared["_order"].astype(int)
    prepared = prepared.drop(columns=["_session_id", "_order"])

    duplicate_columns = prepared.columns[prepared.columns.duplicated()].tolist()
    if duplicate_columns:
        raise ValueError(
            "Arena preparation produced duplicate columns: "
            f"{duplicate_columns}. Available columns: {list(prepared.columns)}"
        )

    base_features = [
        "evaluation_order",
        "history_chars",
        "history_tokens",
        "candidate_a_chars",
        "candidate_b_chars",
        "candidate_a_tokens",
        "candidate_b_tokens",
        "candidate_char_delta",
        "candidate_token_delta",
        "model_a_feature",
        "model_b_feature",
        "category_feature",
    ]
    preference_features = ["pref_a", "pref_b", "pref_tie", "pref_both_bad"]
    return prepared, base_features, preference_features



def print_arena_target_diagnostics(df: pd.DataFrame) -> None:
    """Print direct evidence about whether the vote predicts continuation.

    These diagnostics are intentionally model-free. They reveal whether the raw
    vote categories have any visible association with the future target before
    a classifier is fitted.
    """
    from sklearn.metrics import brier_score_loss, log_loss

    target = "session_continues_after_vote"
    y = df[target].astype(int).to_numpy()
    prevalence = float(np.mean(y))
    constant_probability = np.full(
        len(y),
        np.clip(prevalence, 1e-12, 1.0 - 1e-12),
        dtype=float,
    )

    print_header("Arena target diagnostics")
    print(f"Rows: {len(df):,}")
    print(f"Positive continuation rate: {prevalence:.6f}")
    print(f"Always-stop accuracy: {1.0 - prevalence:.6f}")
    print(
        "Constant-prevalence log loss: "
        f"{log_loss(y, constant_probability, labels=[0, 1]):.6f}"
    )
    print(
        "Constant-prevalence Brier score: "
        f"{brier_score_loss(y, constant_probability):.6f}"
    )

    vote_table = (
        df.groupby("winner_norm", dropna=False)[target]
        .agg(["count", "sum", "mean"])
        .rename(
            columns={
                "sum": "continuations",
                "mean": "continuation_rate",
            }
        )
        .sort_values("continuation_rate", ascending=False)
    )
    vote_table["lift_vs_overall"] = (
        vote_table["continuation_rate"] - prevalence
    )
    print("\nContinuation by current vote")
    print(vote_table.to_string(float_format=lambda value: f"{value:.6f}"))

    minimum_order = int(df["evaluation_order"].min())
    first = df[df["evaluation_order"] == minimum_order]
    if not first.empty:
        first_prevalence = float(first[target].mean())
        first_table = (
            first.groupby("winner_norm", dropna=False)[target]
            .agg(["count", "sum", "mean"])
            .rename(
                columns={
                    "sum": "continuations",
                    "mean": "continuation_rate",
                }
            )
            .sort_values("continuation_rate", ascending=False)
        )
        first_table["lift_vs_first_round"] = (
            first_table["continuation_rate"] - first_prevalence
        )
        print(f"\nContinuation by vote at evaluation_order={minimum_order}")
        print(
            first_table.to_string(
                float_format=lambda value: f"{value:.6f}"
            )
        )

    order_table = (
        df.groupby("evaluation_order")[target]
        .agg(["count", "mean"])
        .rename(columns={"mean": "continuation_rate"})
        .head(12)
    )
    print("\nContinuation by evaluation order")
    print(order_table.to_string(float_format=lambda value: f"{value:.6f}"))

    session_sizes = df.groupby("evaluation_session_id").size()
    print("\nSession completeness summary")
    print(session_sizes.describe().to_string())


def run_arena_matrix(
    args: argparse.Namespace,
    seeds: Sequence[int],
) -> tuple[list[ResultRow], list[SummaryRow]]:
    raw = load_arena_dataframe(
        args.limit_rows,
        sample_seed=args.bootstrap_seed,
    )
    df, base_features, preference_features = prepare_arena_future_dataset(raw)
    print_arena_target_diagnostics(df)

    print_header("Arena dataset audit")
    session_counts = df["evaluation_session_id"].value_counts()
    print(f"Rows: {len(df):,}")
    print(f"Sessions: {df['evaluation_session_id'].nunique():,}")
    print(session_counts.describe().to_string())
    print("Sessions with 2+ evaluations:", int((session_counts >= 2).sum()))
    print("Sessions with 3+ evaluations:", int((session_counts >= 3).sum()))
    print("Sessions with 5+ evaluations:", int((session_counts >= 5).sum()))
    print("\nWinner distribution:")
    print(df["winner_norm"].value_counts(dropna=False).to_string())
    print("\nContinuation target distribution:")
    print(
        df["session_continues_after_vote"]
        .value_counts(normalize=True)
        .to_string()
    )

    condition = "arena"
    condition_rows: list[ResultRow] = []
    condition_stats: list[pd.DataFrame] = []

    for seed in seeds:
        bundle = evaluate_binary_feature_sets(
            df,
            group_column="evaluation_session_id",
            target_column="session_continues_after_vote",
            base_features=base_features,
            preference_features=preference_features,
            track="arena",
            condition=condition,
            seed=seed,
            test_fraction=args.test_fraction,
        )
        condition_rows.extend(bundle.rows)
        condition_stats.append(bundle.session_loss_stats)

    combined_stats = pd.concat(condition_stats, ignore_index=True)
    summaries = build_summary_rows(
        condition_rows,
        combined_stats,
        track="arena",
        condition=condition,
        bootstrap_samples=args.bootstrap_samples,
        confidence_level=args.confidence_level,
        bootstrap_seed=args.bootstrap_seed,
        synthetic_preference_effect=None,
        synthetic_shared_latent_effect=None,
    )
    print_condition_results(condition_rows, summaries)
    return condition_rows, summaries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether preferences contain incremental information "
            "about future outcomes."
        )
    )
    parser.add_argument(
        "--track",
        choices=["synthetic", "arena"],
        default="synthetic",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Fallback single seed when --seeds is omitted.",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated run seeds, for example 1,2,3,4,5.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of complete groups/sessions assigned to test.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Hierarchical paired-bootstrap draws.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=1729,
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=None,
        help="Limit Hugging Face rows for an Arena smoke test.",
    )
    parser.add_argument(
        "--synthetic-sessions",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--synthetic-max-rounds",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--synthetic-preference-effects",
        default="0.75",
        help=(
            "Comma-separated direct preference-to-future coefficients. "
            "Use 0 for the strict null."
        ),
    )
    parser.add_argument(
        "--synthetic-shared-latent-effect",
        type=float,
        default=0.0,
        help=(
            "Optional shared latent effect on the future. Leave at 0 for "
            "the strict null experiment."
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help="CSV path for per-seed feature-set results.",
    )
    parser.add_argument(
        "--summary-out",
        default=None,
        help="CSV path for PFI/bootstrap summaries.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    if not 0.0 < args.test_fraction < 1.0:
        raise SystemExit("--test-fraction must be between 0 and 1.")
    if args.bootstrap_samples < 0:
        raise SystemExit("--bootstrap-samples must be non-negative.")

    seeds = parse_int_list(args.seeds, fallback=args.seed)
    for seed in seeds:
        seed_everything(seed)

    if args.track == "synthetic":
        rows, summaries = run_synthetic_matrix(args, seeds)
    elif args.track == "arena":
        rows, summaries = run_arena_matrix(args, seeds)
    else:
        raise AssertionError(args.track)

    print_header("PreferenceFutures final summary")
    summary_df = pd.DataFrame(
        [dataclasses.asdict(row) for row in summaries]
    )
    print(
        summary_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    if args.out:
        detailed_df = pd.DataFrame(
            [dataclasses.asdict(row) for row in rows]
        )
        detailed_df.to_csv(args.out, index=False)
        print(f"\nSaved per-seed results to {args.out}")

    if args.summary_out:
        summary_df.to_csv(args.summary_out, index=False)
        print(f"Saved summary results to {args.summary_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
