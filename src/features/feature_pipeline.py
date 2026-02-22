from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.config import (
    DEFAULT_FEATURE_ACOUSTIC_PATH,
    DEFAULT_FEATURE_CORE_PATH,
    DEFAULT_FEATURE_MULTIFRACTAL_PATH,
    DEFAULT_FEATURE_SPLITS_PATH,
    DEFAULT_FEATURE_SUMMARY_JSON_PATH,
    DEFAULT_MANIFEST_PATH,
)
from src.data.data_pipeline import PipelineOptions, load_dataset_dataframe


@dataclass(slots=True)
class FeatureOptions:
    prefix: Path | str = Path(".")
    input_manifest: Path = DEFAULT_MANIFEST_PATH
    output_core: Path = DEFAULT_FEATURE_CORE_PATH
    output_acoustic: Path = DEFAULT_FEATURE_ACOUSTIC_PATH
    output_multifractal: Path = DEFAULT_FEATURE_MULTIFRACTAL_PATH
    output_splits: Path = DEFAULT_FEATURE_SPLITS_PATH
    output_summary_json: Path = DEFAULT_FEATURE_SUMMARY_JSON_PATH
    include_splits: bool = True
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    max_samples_per_class: int | None = None
    normalize_audio: bool = True
    target_sample_rate: int | None = None
    mfdfa_order: int = 1
    mfdfa_q_min: float = -5.0
    mfdfa_q_max: float = 5.0
    mfdfa_q_step: float = 1.0
    mfdfa_num_scales: int = 20

    @property
    def prefix_path(self) -> Path:
        return Path(self.prefix)

    def resolve_path(self, path: Path | str) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        return self.prefix_path / candidate

    @property
    def resolved_input_manifest(self) -> Path:
        return self.resolve_path(self.input_manifest)

    @property
    def resolved_output_core(self) -> Path:
        return self.resolve_path(self.output_core)

    @property
    def resolved_output_acoustic(self) -> Path:
        return self.resolve_path(self.output_acoustic)

    @property
    def resolved_output_multifractal(self) -> Path:
        return self.resolve_path(self.output_multifractal)

    @property
    def resolved_output_splits(self) -> Path:
        return self.resolve_path(self.output_splits)

    @property
    def resolved_output_summary_json(self) -> Path:
        return self.resolve_path(self.output_summary_json)


def _feature_build_config(options: FeatureOptions) -> dict[str, Any]:
    return {
        "input_manifest": str(options.resolved_input_manifest),
        "include_splits": bool(options.include_splits),
        "train_ratio": float(options.train_ratio),
        "val_ratio": float(options.val_ratio),
        "test_ratio": float(options.test_ratio),
        "random_seed": int(options.random_seed),
        "max_samples_per_class": options.max_samples_per_class,
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


def _ensure_required_manifest_columns(df: pd.DataFrame) -> None:
    required = ["sample_key", "wav_path", "pathology_de", "pathology_en"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            "Manifest missing required columns: "
            f"{missing}. Available columns={list(df.columns)}"
        )


def _limit_samples_per_class(
    manifest_df: pd.DataFrame, *, max_samples_per_class: int | None, random_seed: int
) -> pd.DataFrame:
    if max_samples_per_class is None:
        return manifest_df

    if max_samples_per_class <= 0:
        raise ValueError("max_samples_per_class must be > 0 when provided.")

    if "pathology_de" in manifest_df.columns:
        class_col = "pathology_de"
    elif "pathology_en" in manifest_df.columns:
        class_col = "pathology_en"
    else:  # pragma: no cover - protected by _ensure_required_manifest_columns
        return manifest_df

    sampled_groups: list[pd.DataFrame] = []
    for _, group in manifest_df.groupby(class_col, dropna=False, sort=True):
        if len(group) <= max_samples_per_class:
            sampled_groups.append(group)
            continue

        sampled_groups.append(
            group.sample(n=max_samples_per_class, random_state=random_seed)
        )

    if not sampled_groups:
        return manifest_df.iloc[0:0].copy()

    return pd.concat(sampled_groups, ignore_index=True)


def _to_float_mono(signal: np.ndarray, normalize: bool) -> np.ndarray:
    arr = np.asarray(signal, dtype=np.float32)
    if arr.ndim > 1:
        arr = np.mean(arr, axis=1)

    if not normalize:
        return arr

    max_abs = float(np.max(np.abs(arr))) if arr.size else 0.0
    if max_abs <= 0.0:
        return arr
    return arr / max_abs


def _load_audio(
    wav_path: Path, *, target_sample_rate: int | None, normalize: bool
) -> dict:
    try:
        signal, sr = librosa.load(
            wav_path,
            sr=target_sample_rate,
            mono=True,
            dtype=np.float32,
        )
        signal = _to_float_mono(signal, normalize=normalize)
        return {
            "status": "ok",
            "error": None,
            "signal": signal,
            "sample_rate": int(sr),
            "num_samples": int(signal.shape[0]),
            "duration_seconds": float(signal.shape[0] / sr) if sr > 0 else 0.0,
        }
    except Exception as exc:  # pragma: no cover - defensive path
        return {
            "status": "failed",
            "error": str(exc),
            "signal": np.array([], dtype=np.float32),
            "sample_rate": None,
            "num_samples": 0,
            "duration_seconds": 0.0,
        }


def _resolve_wav_path(wav_path_raw: str, options: FeatureOptions) -> Path:
    raw_path = Path(wav_path_raw)
    if raw_path.is_absolute():
        return raw_path

    candidates: list[Path] = [
        options.resolve_path(raw_path),
        options.resolved_input_manifest.parent / raw_path,
        Path.cwd() / raw_path,
    ]

    if "data" in raw_path.parts:
        data_idx = raw_path.parts.index("data")
        candidates.append(options.prefix_path / Path(*raw_path.parts[data_idx:]))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def _nan_safe_stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
        }

    return {
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_std": float(np.std(finite)),
        f"{prefix}_min": float(np.min(finite)),
        f"{prefix}_max": float(np.max(finite)),
    }


def _extract_acoustic_features(signal: np.ndarray, sr: int) -> dict[str, Any]:
    if signal.size == 0:
        return {
            "acoustic_status": "empty_signal",
            "acoustic_error": "Audio signal is empty.",
        }

    try:
        y = np.asarray(signal, dtype=np.float32)
        eps = 1e-12

        rms = librosa.feature.rms(y=y).ravel()
        zcr = librosa.feature.zero_crossing_rate(y=y).ravel()
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr).ravel()
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).ravel()
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).ravel()
        flatness = librosa.feature.spectral_flatness(y=y).ravel()

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_delta = librosa.feature.delta(mfcc)

        try:
            f0 = librosa.yin(
                y=y,
                fmin=float(librosa.note_to_hz("C2")),
                fmax=float(librosa.note_to_hz("C7")),
                sr=sr,
            )
        except Exception:
            f0 = np.array([], dtype=float)

        features: dict[str, Any] = {
            "acoustic_status": "ok",
            "acoustic_error": None,
            "ac_time_energy": float(np.mean(y**2)),
            "ac_time_abs_mean": float(np.mean(np.abs(y))),
            "ac_time_peak": float(np.max(np.abs(y))),
            "ac_time_crest_factor": float(
                np.max(np.abs(y)) / (np.sqrt(np.mean(y**2)) + eps)
            ),
        }

        features.update(_nan_safe_stats("ac_rms", rms))
        features.update(_nan_safe_stats("ac_zcr", zcr))
        features.update(_nan_safe_stats("ac_spec_centroid", centroid))
        features.update(_nan_safe_stats("ac_spec_bandwidth", bandwidth))
        features.update(_nan_safe_stats("ac_spec_rolloff", rolloff))
        features.update(_nan_safe_stats("ac_spec_flatness", flatness))

        for idx in range(mfcc.shape[0]):
            coeff = mfcc[idx]
            dcoeff = mfcc_delta[idx]
            features[f"ac_mfcc{idx + 1}_mean"] = float(np.mean(coeff))
            features[f"ac_mfcc{idx + 1}_std"] = float(np.std(coeff))
            features[f"ac_mfcc{idx + 1}_delta_mean"] = float(np.mean(dcoeff))
            features[f"ac_mfcc{idx + 1}_delta_std"] = float(np.std(dcoeff))

        finite_f0 = f0[np.isfinite(f0)] if f0.size else np.array([], dtype=float)
        if finite_f0.size > 0:
            features["ac_f0_mean"] = float(np.mean(finite_f0))
            features["ac_f0_std"] = float(np.std(finite_f0))
            features["ac_f0_min"] = float(np.min(finite_f0))
            features["ac_f0_max"] = float(np.max(finite_f0))
        else:
            features["ac_f0_mean"] = np.nan
            features["ac_f0_std"] = np.nan
            features["ac_f0_min"] = np.nan
            features["ac_f0_max"] = np.nan

        return features
    except Exception as exc:  # pragma: no cover - defensive path
        return {
            "acoustic_status": "failed",
            "acoustic_error": str(exc),
        }


def _compute_scales(num_samples: int, num_scales: int) -> np.ndarray:
    min_scale = 16
    max_scale = max(min_scale + 1, num_samples // 4)
    if max_scale <= min_scale:
        return np.array([], dtype=int)

    scales = np.unique(
        np.logspace(
            np.log10(min_scale),
            np.log10(max_scale),
            num=max(num_scales, 6),
        ).astype(int)
    )
    return scales[scales > 1]


def _estimate_hq(lags: np.ndarray, fq: np.ndarray) -> np.ndarray:
    lags = np.asarray(lags, dtype=float)
    fq = np.asarray(fq, dtype=float)

    if fq.ndim != 2:
        return np.array([], dtype=float)

    if fq.shape[0] != lags.shape[0] and fq.shape[1] == lags.shape[0]:
        fq = fq.T

    if fq.shape[0] != lags.shape[0]:
        return np.array([], dtype=float)

    log_lags = np.log(lags)
    hq_values: list[float] = []

    for i in range(fq.shape[1]):
        curve = fq[:, i]
        valid = np.isfinite(curve) & (curve > 0) & np.isfinite(log_lags)
        if np.count_nonzero(valid) < 3:
            hq_values.append(np.nan)
            continue

        slope, _ = np.polyfit(log_lags[valid], np.log(curve[valid]), deg=1)
        hq_values.append(float(slope))

    return np.asarray(hq_values, dtype=float)


def _extract_multifractal_features(
    signal: np.ndarray, options: FeatureOptions
) -> dict[str, Any]:
    if signal.size == 0:
        return {
            "mf_status": "empty_signal",
            "mf_error": "Audio signal is empty.",
        }

    try:
        from MFDFA import MFDFA as mfdfa
    except Exception as exc:
        return {
            "mf_status": "missing_dependency",
            "mf_error": str(exc),
        }

    try:
        q = np.arange(
            options.mfdfa_q_min,
            options.mfdfa_q_max + options.mfdfa_q_step / 2.0,
            options.mfdfa_q_step,
            dtype=float,
        )
        scales = _compute_scales(
            num_samples=int(signal.shape[0]),
            num_scales=options.mfdfa_num_scales,
        )

        if q.size < 3:
            return {
                "mf_status": "invalid_params",
                "mf_error": "MFDFA requires at least 3 q-values.",
            }
        if scales.size < 6:
            return {
                "mf_status": "short_signal",
                "mf_error": "Insufficient samples for robust MFDFA scales.",
            }

        lags, fq = mfdfa(signal, lag=scales, q=q, order=options.mfdfa_order)
        fq_arr = np.asarray(fq)
        hq = _estimate_hq(np.asarray(lags), fq_arr)

        q_nonzero = q[q != 0.0]
        if hq.size == q.size:
            q_effective = q
        elif hq.size == q_nonzero.size:
            q_effective = q_nonzero
        else:
            return {
                "mf_status": "failed",
                "mf_error": "Unexpected MFDFA output shape for h(q) estimation.",
            }

        tau_q = q_effective * hq - 1.0
        alpha = np.gradient(tau_q, q_effective)
        f_alpha = q_effective * alpha - tau_q

        valid_hq = hq[np.isfinite(hq)]
        valid_tau = tau_q[np.isfinite(tau_q)]
        valid_alpha = alpha[np.isfinite(alpha)]
        valid_falpha = f_alpha[np.isfinite(f_alpha)]

        if valid_alpha.size == 0 or valid_falpha.size == 0:
            return {
                "mf_status": "failed",
                "mf_error": "MFDFA produced non-finite spectrum values.",
            }

        peak_idx = int(np.nanargmax(f_alpha)) if np.any(np.isfinite(f_alpha)) else 0
        alpha_min = float(np.nanmin(alpha))
        alpha_max = float(np.nanmax(alpha))
        alpha_peak = float(alpha[peak_idx]) if np.isfinite(alpha[peak_idx]) else np.nan
        width = float(alpha_max - alpha_min)
        eps = 1e-12
        asym = (
            float((alpha_peak - alpha_min) / (alpha_max - alpha_peak + eps))
            if np.isfinite(alpha_peak)
            else np.nan
        )

        return {
            "mf_status": "ok",
            "mf_error": None,
            "mf_hq_mean": float(np.nanmean(valid_hq)) if valid_hq.size else np.nan,
            "mf_hq_std": float(np.nanstd(valid_hq)) if valid_hq.size else np.nan,
            "mf_hq_min": float(np.nanmin(valid_hq)) if valid_hq.size else np.nan,
            "mf_hq_max": float(np.nanmax(valid_hq)) if valid_hq.size else np.nan,
            "mf_tau_mean": float(np.nanmean(valid_tau)) if valid_tau.size else np.nan,
            "mf_tau_std": float(np.nanstd(valid_tau)) if valid_tau.size else np.nan,
            "mf_alpha_mean": float(np.nanmean(valid_alpha))
            if valid_alpha.size
            else np.nan,
            "mf_alpha_std": float(np.nanstd(valid_alpha))
            if valid_alpha.size
            else np.nan,
            "mf_spectrum_width": width,
            "mf_spectrum_peak_alpha": alpha_peak,
            "mf_spectrum_peak_f": float(f_alpha[peak_idx])
            if np.isfinite(f_alpha[peak_idx])
            else np.nan,
            "mf_spectrum_asymmetry": asym,
            "mf_num_scales": int(scales.size),
            "mf_num_q": int(q_effective.size),
        }
    except Exception as exc:  # pragma: no cover - defensive path
        return {
            "mf_status": "failed",
            "mf_error": str(exc),
        }


def _build_random_split_table(
    sample_keys: list[str], options: FeatureOptions
) -> pd.DataFrame:
    if not sample_keys:
        return pd.DataFrame(columns=["sample_key", "split", "split_seed"])

    total = options.train_ratio + options.val_ratio + options.test_ratio
    if total <= 0:
        raise ValueError("train_ratio + val_ratio + test_ratio must be > 0.")

    train_ratio = options.train_ratio / total
    val_ratio = options.val_ratio / total

    rng = np.random.default_rng(options.random_seed)
    indices = np.arange(len(sample_keys))
    rng.shuffle(indices)

    n_total = len(sample_keys)
    n_train = int(np.floor(n_total * train_ratio))
    n_val = int(np.floor(n_total * val_ratio))

    split_values = np.full(n_total, "test", dtype=object)
    split_values[indices[:n_train]] = "train"
    split_values[indices[n_train : n_train + n_val]] = "val"

    return pd.DataFrame(
        {
            "sample_key": sample_keys,
            "split": split_values,
            "split_seed": options.random_seed,
        }
    )


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


def _prepare_target_manifest(options: FeatureOptions) -> pd.DataFrame:
    base_pipeline_options = PipelineOptions(prefix=options.prefix)
    manifest_df = load_dataset_dataframe(
        manifest_path=options.input_manifest,
        build_if_missing=True,
        options=base_pipeline_options,
        save_if_built=True,
    )
    _ensure_required_manifest_columns(manifest_df)
    return _limit_samples_per_class(
        manifest_df,
        max_samples_per_class=options.max_samples_per_class,
        random_seed=options.random_seed,
    )


def _extract_feature_tables_from_manifest(
    manifest_df: pd.DataFrame, options: FeatureOptions
) -> dict[str, pd.DataFrame]:
    if manifest_df.empty:
        tables: dict[str, pd.DataFrame] = {
            "core": pd.DataFrame(),
            "acoustic": pd.DataFrame(),
            "multifractal": pd.DataFrame(),
        }
        if options.include_splits:
            tables["splits"] = pd.DataFrame(
                columns=["sample_key", "split", "split_seed"]
            )
        return tables

    meta_columns = [
        "sample_key",
        "duplicate_class_key",
        "recording_id",
        "speaker_id",
        "pathology_de",
        "pathology_en",
        "is_healthy",
        "modality",
        "token",
        "sex",
        "is_overlap_speaker",
        "is_overlap_speaker_id",
        "wav_path",
    ]
    available_meta = [col for col in meta_columns if col in manifest_df.columns]

    core_rows: list[dict[str, Any]] = []
    acoustic_rows: list[dict[str, Any]] = []
    multifractal_rows: list[dict[str, Any]] = []

    records = manifest_df.to_dict(orient="records")
    for row in tqdm(records, desc="Extracting features", unit="sample"):
        sample_key = str(row["sample_key"])
        wav_path_raw = row.get("wav_path")
        base_meta = {col: row.get(col) for col in available_meta}

        if not isinstance(wav_path_raw, str) or not wav_path_raw.strip():
            core_rows.append(
                {
                    **base_meta,
                    "sample_key": sample_key,
                    "feature_status": "missing_wav_path",
                    "feature_error": "wav_path missing in manifest.",
                    "audio_sample_rate": None,
                    "audio_num_samples": 0,
                    "audio_duration_seconds": 0.0,
                }
            )
            acoustic_rows.append(
                {
                    "sample_key": sample_key,
                    "acoustic_status": "missing_wav_path",
                    "acoustic_error": "wav_path missing in manifest.",
                }
            )
            multifractal_rows.append(
                {
                    "sample_key": sample_key,
                    "mf_status": "missing_wav_path",
                    "mf_error": "wav_path missing in manifest.",
                }
            )
            continue

        wav_path = _resolve_wav_path(wav_path_raw, options=options)
        if not wav_path.exists():
            core_rows.append(
                {
                    **base_meta,
                    "sample_key": sample_key,
                    "feature_status": "missing_wav_file",
                    "feature_error": f"WAV file not found: {wav_path}",
                    "audio_sample_rate": None,
                    "audio_num_samples": 0,
                    "audio_duration_seconds": 0.0,
                }
            )
            acoustic_rows.append(
                {
                    "sample_key": sample_key,
                    "acoustic_status": "missing_wav_file",
                    "acoustic_error": f"WAV file not found: {wav_path}",
                }
            )
            multifractal_rows.append(
                {
                    "sample_key": sample_key,
                    "mf_status": "missing_wav_file",
                    "mf_error": f"WAV file not found: {wav_path}",
                }
            )
            continue

        audio_info = _load_audio(
            wav_path,
            target_sample_rate=options.target_sample_rate,
            normalize=options.normalize_audio,
        )

        core_row = {
            **base_meta,
            "sample_key": sample_key,
            "feature_status": audio_info["status"],
            "feature_error": audio_info["error"],
            "audio_sample_rate": audio_info["sample_rate"],
            "audio_num_samples": audio_info["num_samples"],
            "audio_duration_seconds": audio_info["duration_seconds"],
        }

        if audio_info["status"] != "ok":
            acoustic_rows.append(
                {
                    "sample_key": sample_key,
                    "acoustic_status": "audio_load_failed",
                    "acoustic_error": audio_info["error"],
                }
            )
            multifractal_rows.append(
                {
                    "sample_key": sample_key,
                    "mf_status": "audio_load_failed",
                    "mf_error": audio_info["error"],
                }
            )
            core_rows.append(core_row)
            continue

        signal = np.asarray(audio_info["signal"], dtype=np.float32)
        sr = int(audio_info["sample_rate"])

        acoustic_features = _extract_acoustic_features(signal, sr=sr)
        mf_features = _extract_multifractal_features(signal, options=options)

        acoustic_rows.append({"sample_key": sample_key, **acoustic_features})
        multifractal_rows.append({"sample_key": sample_key, **mf_features})

        if (
            acoustic_features.get("acoustic_status") != "ok"
            or mf_features.get("mf_status") != "ok"
        ):
            core_row["feature_status"] = "partial_failure"
            errors = [
                str(acoustic_features.get("acoustic_error") or ""),
                str(mf_features.get("mf_error") or ""),
            ]
            merged_error = " | ".join([e for e in errors if e.strip()])
            core_row["feature_error"] = merged_error or None

        core_rows.append(core_row)

    core_df = pd.DataFrame(core_rows)
    acoustic_df = pd.DataFrame(acoustic_rows)
    multifractal_df = pd.DataFrame(multifractal_rows)

    tables: dict[str, pd.DataFrame] = {
        "core": core_df,
        "acoustic": acoustic_df,
        "multifractal": multifractal_df,
    }

    if options.include_splits:
        split_df = _build_random_split_table(
            sample_keys=core_df["sample_key"].astype(str).tolist(),
            options=options,
        )
        tables["splits"] = split_df

    return tables


def extract_feature_tables(options: FeatureOptions) -> dict[str, pd.DataFrame]:
    manifest_df = _prepare_target_manifest(options)
    return _extract_feature_tables_from_manifest(manifest_df, options)


def _dedupe_by_sample_key(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "sample_key" not in df.columns:
        return df
    return df.drop_duplicates(subset=["sample_key"], keep="last")


def _load_existing_feature_tables(options: FeatureOptions) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {
        "core": _read_dataframe(options.resolved_output_core),
        "acoustic": _read_dataframe(options.resolved_output_acoustic),
        "multifractal": _read_dataframe(options.resolved_output_multifractal),
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

    tables: dict[str, pd.DataFrame] = {
        "core": core,
        "acoustic": acoustic,
        "multifractal": multifractal,
    }

    if options.include_splits:
        tables["splits"] = _build_random_split_table(
            sample_keys=core["sample_key"].astype(str).tolist()
            if "sample_key" in core.columns
            else [],
            options=options,
        )

    return tables


def _tables_have_exact_target_keys(
    tables: dict[str, pd.DataFrame], target_keys: set[str]
) -> bool:
    for table_name in ("core", "acoustic", "multifractal"):
        if (
            _existing_sample_key_set(tables.get(table_name, pd.DataFrame()))
            != target_keys
        ):
            return False
    return True


def build_feature_tables(options: FeatureOptions) -> dict[str, pd.DataFrame]:
    """Backward-compatible alias for extract_feature_tables."""
    return extract_feature_tables(options)


def save_feature_tables(
    tables: dict[str, pd.DataFrame], options: FeatureOptions
) -> None:
    _write_dataframe(tables["core"], options.resolved_output_core)
    _write_dataframe(tables["acoustic"], options.resolved_output_acoustic)
    _write_dataframe(tables["multifractal"], options.resolved_output_multifractal)

    if options.include_splits and "splits" in tables:
        _write_dataframe(tables["splits"], options.resolved_output_splits)

    _save_feature_build_config(options)


def summarize_feature_tables(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    core_df = tables.get("core", pd.DataFrame())
    acoustic_df = tables.get("acoustic", pd.DataFrame())
    multifractal_df = tables.get("multifractal", pd.DataFrame())

    summary = {
        "num_samples": int(len(core_df)),
        "num_acoustic_rows": int(len(acoustic_df)),
        "num_multifractal_rows": int(len(multifractal_df)),
        "feature_status_counts": {},
        "acoustic_status_counts": {},
        "multifractal_status_counts": {},
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
    splits_path = effective_options.resolved_output_splits

    core_exists = core_path.exists()
    acoustic_exists = acoustic_path.exists()
    multifractal_exists = multifractal_path.exists()
    splits_exists = (not effective_options.include_splits) or splits_path.exists()

    if core_exists and acoustic_exists and multifractal_exists and splits_exists:
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

            tables: dict[str, pd.DataFrame] = {
                "core": merged_core,
                "acoustic": merged_acoustic,
                "multifractal": merged_multifractal,
            }

            if effective_options.include_splits:
                tables["splits"] = _build_random_split_table(
                    sample_keys=merged_core["sample_key"].astype(str).tolist(),
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
