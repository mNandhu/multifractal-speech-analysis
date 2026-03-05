"""Sanity-check display utilities for the model training pipeline.

All functions here **display** diagnostic information — they do not assert or
raise.  Call them after loading / merging data to visually confirm everything
looks reasonable before training.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# ── helpers ──────────────────────────────────────────────────────────────────


def _section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def _pct(n: int, total: int) -> str:
    return f"{n} ({100 * n / total:.1f}%)" if total else str(n)


# ── public API ───────────────────────────────────────────────────────────────


def display_data_overview(
    df: pd.DataFrame,
    *,
    target_col: str = "target_label",
    speaker_col: str = "speaker_id",
    token_col: str = "token",
) -> None:
    """High-level dataset shape, class distribution, and speaker counts."""
    _section("Data Overview")
    print(f"Total samples (rows):  {len(df)}")
    print(f"Total columns:         {df.shape[1]}")

    if speaker_col in df.columns:
        print(f"Unique speakers:       {df[speaker_col].nunique()}")

    if token_col in df.columns:
        tokens = df[token_col].unique()
        print(f"Unique tokens:         {len(tokens)}  →  {sorted(tokens)}")

    if "sex" in df.columns:
        print(f"Sex distribution:      {df['sex'].value_counts().to_dict()}")

    if target_col in df.columns:
        _section("Class Distribution")
        counts = df[target_col].value_counts()
        for cls, n in counts.items():
            print(f"  {str(cls):30s}  {_pct(n, len(df))}")

    if "is_healthy" in df.columns:
        n_h = int(df["is_healthy"].astype(int).sum())
        n_p = len(df) - n_h
        print(
            f"\n  Binary balance → Healthy: {n_h}  |  Pathological: {n_p}  |  ratio: {n_h / max(n_p, 1):.2f}"
        )


def display_audio_properties(df: pd.DataFrame) -> None:
    """Show sample rate, duration, and sample count stats."""
    _section("Audio Properties")

    if "audio_sample_rate" in df.columns:
        sr = df["audio_sample_rate"]
        unique_sr = sr.dropna().unique()
        if len(unique_sr) == 1:
            print(f"Sample rate:  {int(unique_sr[0])} Hz  (uniform ✓)")
        else:
            print(f"Sample rate:  NOT uniform ✗  →  {sorted(unique_sr)}")
            print(f"  counts: {sr.value_counts().to_dict()}")
    else:
        print("Sample rate:  column 'audio_sample_rate' not found")

    if "audio_duration_seconds" in df.columns:
        dur = df["audio_duration_seconds"].dropna()
        print(
            f"Duration (s): mean={dur.mean():.2f}  std={dur.std():.2f}  "
            f"min={dur.min():.2f}  max={dur.max():.2f}"
        )
    if "audio_num_samples" in df.columns:
        ns = df["audio_num_samples"].dropna()
        print(
            f"Num samples:  mean={ns.mean():.0f}  min={ns.min():.0f}  max={ns.max():.0f}"
        )


def display_feature_summary(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> None:
    """Show loaded feature groups, NaN rates, and basic stats."""
    _section("Feature Summary")

    # Group features by prefix
    groups: dict[str, list[str]] = {}
    for c in numeric_cols:
        prefix = c.split("_")[0] if "_" in c else "other"
        groups.setdefault(prefix, []).append(c)

    print(f"Total numeric features: {len(numeric_cols)}")
    print(f"Categorical features:   {categorical_cols or '(none)'}")
    print()
    print(f"  {'Group':<12s}  {'Count':>5s}  {'Example columns'}")
    print(f"  {'─' * 12}  {'─' * 5}  {'─' * 40}")
    for prefix in sorted(groups):
        cols = groups[prefix]
        examples = ", ".join(cols[:3])
        if len(cols) > 3:
            examples += f", … ({len(cols) - 3} more)"
        print(f"  {prefix:<12s}  {len(cols):>5d}  {examples}")

    # NaN overview
    _section("Missing Values (NaN)")
    nan_counts = df[numeric_cols].isna().sum()
    n_with_nan = int((nan_counts > 0).sum())
    print(f"Features with any NaN:  {n_with_nan} / {len(numeric_cols)}")

    if n_with_nan > 0:
        top_nan = nan_counts[nan_counts > 0].sort_values(ascending=False).head(10)
        print(f"\n  Top NaN features (of {n_with_nan}):")
        for col, cnt in top_nan.items():
            print(f"    {col:40s}  {_pct(int(cnt), len(df))}")


def display_feature_status(
    tables: dict[str, pd.DataFrame],
) -> None:
    """Show extraction success/failure rates per feature table."""
    status_cols = {
        "core": "feature_status",
        "acoustic": "acoustic_status",
        "multifractal": "mf_status",
        "opensmile": "opensmile_status",
    }

    _section("Feature Extraction Status")
    for table_name, status_col in status_cols.items():
        tbl = tables.get(table_name)
        if tbl is None or tbl.empty:
            print(f"  {table_name:15s}  (not loaded)")
            continue
        if status_col not in tbl.columns:
            print(f"  {table_name:15s}  {len(tbl)} rows  (no status column)")
            continue
        counts = tbl[status_col].value_counts().to_dict()
        print(f"  {table_name:15s}  {len(tbl)} rows  →  {counts}")


def display_build_config(opts: Any) -> None:
    """Show key build-time options that affect the feature set."""
    _section("Build Configuration")
    fields = [
        ("selected_token", opts.selected_token),
        ("target_sample_rate", opts.target_sample_rate),
        ("max_samples_per_class", opts.max_samples_per_class),
        ("balance_healthy", opts.balance_healthy),
        ("normalize_audio", opts.normalize_audio),
        ("mfdfa_order", opts.mfdfa_order),
        (
            "mfdfa_q_range",
            f"[{opts.mfdfa_q_min}, {opts.mfdfa_q_max}] step {opts.mfdfa_q_step}",
        ),
        ("mfdfa_num_scales", opts.mfdfa_num_scales),
        ("random_seed", opts.random_seed),
    ]
    for name, val in fields:
        print(f"  {name:30s}  {val}")


def run_all(
    df: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    opts: Any,
    numeric_cols: list[str],
    categorical_cols: list[str],
    *,
    target_col: str = "target_label",
) -> None:
    """Run every sanity-check display in order."""
    display_build_config(opts)
    display_data_overview(df, target_col=target_col)
    display_audio_properties(df)
    display_feature_status(tables)
    display_feature_summary(df, numeric_cols, categorical_cols)
