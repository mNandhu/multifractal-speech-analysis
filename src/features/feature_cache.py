from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.features.feature_extraction import (
    _build_random_split_table,
    _extract_feature_tables_from_manifest,
    _prepare_target_manifest,
    extract_feature_tables,
)
from src.features.feature_options import FeatureOptions


def _feature_build_config(options: FeatureOptions) -> dict[str, Any]:
    return {
        "input_manifest": str(options.resolved_input_manifest),
        "include_splits": bool(options.include_splits),
        "train_ratio": float(options.train_ratio),
        "val_ratio": float(options.val_ratio),
        "test_ratio": float(options.test_ratio),
        "random_seed": int(options.random_seed),
        "num_workers": options.num_workers,
        "max_samples_per_class": options.max_samples_per_class,
        "balance_healthy": bool(options.balance_healthy),
        "selected_token": options.selected_token,
        "normalize_audio": bool(options.normalize_audio),
        "target_sample_rate": options.target_sample_rate,
        "mfdfa_order": int(options.mfdfa_order),
        "mfdfa_q_min": float(options.mfdfa_q_min),
        "mfdfa_q_max": float(options.mfdfa_q_max),
        "mfdfa_q_step": float(options.mfdfa_q_step),
        "mfdfa_num_scales": int(options.mfdfa_num_scales),
    }


def _feature_build_config_path(options: FeatureOptions) -> Path:
    return options.resolved_output_summary_json.with_name("feature_build_config.json")


def _save_feature_build_config(options: FeatureOptions) -> None:
    path = _feature_build_config_path(options)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_feature_build_config(options), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _load_feature_build_config(options: FeatureOptions) -> dict[str, Any] | None:
    path = _feature_build_config_path(options)
    if not path.exists():
        return None

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return raw
    except Exception:  # pragma: no cover - defensive path
        return None
    return None


def _read_dataframe(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file extension: {path}")


def _write_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
        return
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
        return
    raise ValueError(f"Unsupported file extension: {path}")


def _dedupe_by_sample_key(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "sample_key" not in df.columns:
        return df
    return df.drop_duplicates(subset=["sample_key"], keep="last")


def _load_existing_feature_tables(options: FeatureOptions) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {
        "core": _read_dataframe(options.resolved_output_core),
        "acoustic": _read_dataframe(options.resolved_output_acoustic),
        "multifractal": _read_dataframe(options.resolved_output_multifractal),
        "opensmile": _read_dataframe(options.resolved_output_opensmile),
        "neurokit2": _read_dataframe(options.resolved_output_neurokit2)
        if options.resolved_output_neurokit2.exists()
        else pd.DataFrame(),
    }
    if options.include_splits and options.resolved_output_splits.exists():
        tables["splits"] = _read_dataframe(options.resolved_output_splits)
    return tables


def _same_config_except_max(
    saved_config: dict[str, Any], requested_config: dict[str, Any]
) -> bool:
    left = dict(saved_config)
    right = dict(requested_config)
    left.pop("max_samples_per_class", None)
    right.pop("max_samples_per_class", None)
    return left == right


def _target_sample_key_set(manifest_df: pd.DataFrame) -> set[str]:
    if "sample_key" not in manifest_df.columns:
        return set()
    return set(manifest_df["sample_key"].astype(str))


def _existing_sample_key_set(core_df: pd.DataFrame) -> set[str]:
    if "sample_key" not in core_df.columns:
        return set()
    return set(core_df["sample_key"].astype(str))


def _filter_table_to_target_keys(
    df: pd.DataFrame, target_keys: set[str]
) -> pd.DataFrame:
    if df.empty or "sample_key" not in df.columns:
        return df.copy()
    filtered = df[df["sample_key"].astype(str).isin(target_keys)].copy()
    return _dedupe_by_sample_key(filtered)


def _subset_cached_tables_to_target_keys(
    cached_tables: dict[str, pd.DataFrame],
    target_keys: set[str],
    options: FeatureOptions,
) -> dict[str, pd.DataFrame]:
    core = _filter_table_to_target_keys(
        cached_tables.get("core", pd.DataFrame()), target_keys
    )
    acoustic = _filter_table_to_target_keys(
        cached_tables.get("acoustic", pd.DataFrame()), target_keys
    )
    multifractal = _filter_table_to_target_keys(
        cached_tables.get("multifractal", pd.DataFrame()), target_keys
    )
    opensmile = _filter_table_to_target_keys(
        cached_tables.get("opensmile", pd.DataFrame()), target_keys
    )
    neurokit2 = _filter_table_to_target_keys(
        cached_tables.get("neurokit2", pd.DataFrame()), target_keys
    )

    tables: dict[str, pd.DataFrame] = {
        "core": core,
        "acoustic": acoustic,
        "multifractal": multifractal,
        "opensmile": opensmile,
        "neurokit2": neurokit2,
    }

    if options.include_splits:
        tables["splits"] = _build_random_split_table(
            core_df=core,
            options=options,
        )

    return tables


def _tables_have_exact_target_keys(
    tables: dict[str, pd.DataFrame], target_keys: set[str]
) -> bool:
    for table_name in ("core", "acoustic", "multifractal", "opensmile", "neurokit2"):
        if (
            _existing_sample_key_set(tables.get(table_name, pd.DataFrame()))
            != target_keys
        ):
            return False
    return True


def save_feature_tables(
    tables: dict[str, pd.DataFrame], options: FeatureOptions
) -> None:
    _write_dataframe(tables["core"], options.resolved_output_core)
    _write_dataframe(tables["acoustic"], options.resolved_output_acoustic)
    _write_dataframe(tables["multifractal"], options.resolved_output_multifractal)
    _write_dataframe(tables["opensmile"], options.resolved_output_opensmile)
    if "neurokit2" in tables:
        _write_dataframe(tables["neurokit2"], options.resolved_output_neurokit2)

    if options.include_splits and "splits" in tables:
        _write_dataframe(tables["splits"], options.resolved_output_splits)

    _save_feature_build_config(options)


def summarize_feature_tables(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    core_df = tables.get("core", pd.DataFrame())
    acoustic_df = tables.get("acoustic", pd.DataFrame())
    multifractal_df = tables.get("multifractal", pd.DataFrame())
    opensmile_df = tables.get("opensmile", pd.DataFrame())
    neurokit2_df = tables.get("neurokit2", pd.DataFrame())

    summary = {
        "num_samples": int(len(core_df)),
        "num_acoustic_rows": int(len(acoustic_df)),
        "num_multifractal_rows": int(len(multifractal_df)),
        "num_opensmile_rows": int(len(opensmile_df)),
        "num_neurokit2_rows": int(len(neurokit2_df)),
        "feature_status_counts": {},
        "acoustic_status_counts": {},
        "multifractal_status_counts": {},
        "opensmile_status_counts": {},
        "neurokit2_status_counts": {},
    }

    if "feature_status" in core_df.columns:
        summary["feature_status_counts"] = (
            core_df["feature_status"].value_counts(dropna=False).to_dict()
        )

    if "acoustic_status" in acoustic_df.columns:
        summary["acoustic_status_counts"] = (
            acoustic_df["acoustic_status"].value_counts(dropna=False).to_dict()
        )

    if "mf_status" in multifractal_df.columns:
        summary["multifractal_status_counts"] = (
            multifractal_df["mf_status"].value_counts(dropna=False).to_dict()
        )

    if "opensmile_status" in opensmile_df.columns:
        summary["opensmile_status_counts"] = (
            opensmile_df["opensmile_status"].value_counts(dropna=False).to_dict()
        )

    if "nk_status" in neurokit2_df.columns:
        summary["neurokit2_status_counts"] = (
            neurokit2_df["nk_status"].value_counts(dropna=False).to_dict()
        )

    if "splits" in tables and not tables["splits"].empty:
        summary["split_counts"] = tables["splits"]["split"].value_counts().to_dict()

    return summary


def save_feature_summary_json(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def load_feature_tables(
    options: FeatureOptions | None = None,
    *,
    build_if_missing: bool = True,
    save_if_built: bool = True,
) -> dict[str, pd.DataFrame]:
    effective_options = options or FeatureOptions()

    core_path = effective_options.resolved_output_core
    acoustic_path = effective_options.resolved_output_acoustic
    multifractal_path = effective_options.resolved_output_multifractal
    opensmile_path = effective_options.resolved_output_opensmile
    neurokit2_path = effective_options.resolved_output_neurokit2
    splits_path = effective_options.resolved_output_splits

    core_exists = core_path.exists()
    acoustic_exists = acoustic_path.exists()
    multifractal_exists = multifractal_path.exists()
    opensmile_exists = opensmile_path.exists()
    neurokit2_exists = neurokit2_path.exists()
    splits_exists = (not effective_options.include_splits) or splits_path.exists()

    if (
        core_exists
        and acoustic_exists
        and multifractal_exists
        and opensmile_exists
        and neurokit2_exists
        and splits_exists
    ):
        cached_tables = _load_existing_feature_tables(effective_options)

        if not build_if_missing:
            return cached_tables

        target_manifest = _prepare_target_manifest(effective_options)
        target_keys = _target_sample_key_set(target_manifest)
        existing_keys = _existing_sample_key_set(
            cached_tables.get("core", pd.DataFrame())
        )

        requested_config = _feature_build_config(effective_options)
        saved_config = _load_feature_build_config(effective_options)
        config_matches = saved_config == requested_config if saved_config else False

        # Fast path: cache exactly matches requested build target.
        if config_matches and existing_keys == target_keys:
            return cached_tables

        # Cache downscale path: same config except max_samples_per_class,
        # and requested keys are a strict subset of existing cached keys.
        can_cache_subset = (
            saved_config is not None
            and _same_config_except_max(saved_config, requested_config)
            and target_keys.issubset(existing_keys)
            and len(target_keys) < len(existing_keys)
            and "sample_key" in target_manifest.columns
        )

        if can_cache_subset:
            tables = _subset_cached_tables_to_target_keys(
                cached_tables, target_keys, effective_options
            )

            if _tables_have_exact_target_keys(tables, target_keys):
                if save_if_built:
                    save_feature_tables(tables=tables, options=effective_options)
                    summary = summarize_feature_tables(tables)
                    save_feature_summary_json(
                        summary, effective_options.resolved_output_summary_json
                    )
                return tables

        # Incremental expansion path: same config except max_samples_per_class,
        # and requested target is a strict superset of existing rows.
        can_incremental_expand = (
            saved_config is not None
            and _same_config_except_max(saved_config, requested_config)
            and existing_keys.issubset(target_keys)
            and len(target_keys) > len(existing_keys)
            and "sample_key" in target_manifest.columns
        )

        if can_incremental_expand:
            missing_keys = target_keys - existing_keys
            missing_manifest = target_manifest[
                target_manifest["sample_key"].astype(str).isin(missing_keys)
            ].copy()

            new_tables = _extract_feature_tables_from_manifest(
                missing_manifest, effective_options
            )

            merged_core = _dedupe_by_sample_key(
                pd.concat(
                    [cached_tables.get("core", pd.DataFrame()), new_tables["core"]],
                    ignore_index=True,
                )
            )
            merged_acoustic = _dedupe_by_sample_key(
                pd.concat(
                    [
                        cached_tables.get("acoustic", pd.DataFrame()),
                        new_tables["acoustic"],
                    ],
                    ignore_index=True,
                )
            )
            merged_multifractal = _dedupe_by_sample_key(
                pd.concat(
                    [
                        cached_tables.get("multifractal", pd.DataFrame()),
                        new_tables["multifractal"],
                    ],
                    ignore_index=True,
                )
            )
            merged_opensmile = _dedupe_by_sample_key(
                pd.concat(
                    [
                        cached_tables.get("opensmile", pd.DataFrame()),
                        new_tables["opensmile"],
                    ],
                    ignore_index=True,
                )
            )
            merged_neurokit2 = _dedupe_by_sample_key(
                pd.concat(
                    [
                        cached_tables.get("neurokit2", pd.DataFrame()),
                        new_tables["neurokit2"],
                    ],
                    ignore_index=True,
                )
            )

            tables: dict[str, pd.DataFrame] = {
                "core": merged_core,
                "acoustic": merged_acoustic,
                "multifractal": merged_multifractal,
                "opensmile": merged_opensmile,
                "neurokit2": merged_neurokit2,
            }

            if effective_options.include_splits:
                tables["splits"] = _build_random_split_table(
                    core_df=merged_core,
                    options=effective_options,
                )

            if _tables_have_exact_target_keys(tables, target_keys):
                if save_if_built:
                    save_feature_tables(tables=tables, options=effective_options)
                    summary = summarize_feature_tables(tables)
                    save_feature_summary_json(
                        summary, effective_options.resolved_output_summary_json
                    )

                return tables

        # General reconciliation path: same config family but target keys changed
        # in a non-subset/non-superset way (e.g., sampling semantics update).
        can_reconcile = (
            saved_config is not None
            and _same_config_except_max(saved_config, requested_config)
            and "sample_key" in target_manifest.columns
        )

        if can_reconcile:
            missing_keys = target_keys - existing_keys

            if missing_keys:
                missing_manifest = target_manifest[
                    target_manifest["sample_key"].astype(str).isin(missing_keys)
                ].copy()
                new_tables = _extract_feature_tables_from_manifest(
                    missing_manifest, effective_options
                )
            else:
                new_tables = {
                    "core": pd.DataFrame(),
                    "acoustic": pd.DataFrame(),
                    "multifractal": pd.DataFrame(),
                    "opensmile": pd.DataFrame(),
                    "neurokit2": pd.DataFrame(),
                }

            merged_core = _dedupe_by_sample_key(
                pd.concat(
                    [cached_tables.get("core", pd.DataFrame()), new_tables["core"]],
                    ignore_index=True,
                )
            )
            merged_acoustic = _dedupe_by_sample_key(
                pd.concat(
                    [
                        cached_tables.get("acoustic", pd.DataFrame()),
                        new_tables["acoustic"],
                    ],
                    ignore_index=True,
                )
            )
            merged_multifractal = _dedupe_by_sample_key(
                pd.concat(
                    [
                        cached_tables.get("multifractal", pd.DataFrame()),
                        new_tables["multifractal"],
                    ],
                    ignore_index=True,
                )
            )
            merged_opensmile = _dedupe_by_sample_key(
                pd.concat(
                    [
                        cached_tables.get("opensmile", pd.DataFrame()),
                        new_tables["opensmile"],
                    ],
                    ignore_index=True,
                )
            )
            merged_neurokit2 = _dedupe_by_sample_key(
                pd.concat(
                    [
                        cached_tables.get("neurokit2", pd.DataFrame()),
                        new_tables["neurokit2"],
                    ],
                    ignore_index=True,
                )
            )

            # Keep only requested target keys (drops obsolete rows without recompute).
            merged_core = _filter_table_to_target_keys(merged_core, target_keys)
            merged_acoustic = _filter_table_to_target_keys(merged_acoustic, target_keys)
            merged_multifractal = _filter_table_to_target_keys(
                merged_multifractal, target_keys
            )
            merged_opensmile = _filter_table_to_target_keys(
                merged_opensmile, target_keys
            )
            merged_neurokit2 = _filter_table_to_target_keys(
                merged_neurokit2, target_keys
            )

            tables = {
                "core": merged_core,
                "acoustic": merged_acoustic,
                "multifractal": merged_multifractal,
                "opensmile": merged_opensmile,
                "neurokit2": merged_neurokit2,
            }

            if effective_options.include_splits:
                tables["splits"] = _build_random_split_table(
                    core_df=merged_core,
                    options=effective_options,
                )

            if _tables_have_exact_target_keys(tables, target_keys):
                if save_if_built:
                    save_feature_tables(tables=tables, options=effective_options)
                    summary = summarize_feature_tables(tables)
                    save_feature_summary_json(
                        summary, effective_options.resolved_output_summary_json
                    )
                return tables

        # Any other mismatch is rebuilt to avoid silent stale-cache reuse.
        tables = _extract_feature_tables_from_manifest(
            target_manifest, effective_options
        )
        if save_if_built:
            save_feature_tables(tables=tables, options=effective_options)
            summary = summarize_feature_tables(tables)
            save_feature_summary_json(
                summary, effective_options.resolved_output_summary_json
            )

        return tables

    if not build_if_missing:
        missing = [
            str(p)
            for p, exists in [
                (core_path, core_exists),
                (acoustic_path, acoustic_exists),
                (multifractal_path, multifractal_exists),
                (opensmile_path, opensmile_exists),
                (neurokit2_path, neurokit2_exists),
                (splits_path, splits_exists),
            ]
            if not exists
        ]
        raise FileNotFoundError(
            "Feature table files missing: "
            f"{missing}. Set build_if_missing=True to generate them."
        )

    tables = extract_feature_tables(effective_options)
    if save_if_built:
        save_feature_tables(tables=tables, options=effective_options)
        summary = summarize_feature_tables(tables)
        save_feature_summary_json(
            summary, effective_options.resolved_output_summary_json
        )

    return tables
