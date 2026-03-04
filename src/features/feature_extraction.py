from __future__ import annotations

import importlib.util
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.data.data_pipeline import PipelineOptions, load_dataset_dataframe
from src.features.feature_options import FeatureOptions

# This block looks complicated, but then we are ust checking if imports are present, and if not
# we handle it with grace :)
# Optional heavy dependencies — availability is checked once at import time.
_MFDFA_AVAILABLE = importlib.util.find_spec("MFDFA") is not None
_OPENSMILE_AVAILABLE = importlib.util.find_spec("opensmile") is not None

_mfdfa_func: Any = None
_opensmile_mod: Any = None
_SMILE_SINGLETON: Any = None

if _MFDFA_AVAILABLE:
    from MFDFA import MFDFA as _mfdfa_func  # type: ignore[import-untyped]

if _OPENSMILE_AVAILABLE:
    import opensmile as _opensmile_mod  # type: ignore[import-untyped]


def _ensure_required_manifest_columns(df: pd.DataFrame) -> None:
    """Raise ValueError if any required manifest columns are missing."""
    required = ["sample_key", "wav_path", "pathology_de", "pathology_en"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            "Manifest missing required columns: "
            f"{missing}. Available columns={list(df.columns)}"
        )


def _limit_samples_per_class(
    manifest_df: pd.DataFrame,
    *,
    max_samples_per_class: int | None,
    random_seed: int,
    skip_healthy: bool = False,
) -> pd.DataFrame:
    """Randomly downsample each pathology class to at most *max_samples_per_class* rows.

    Classes with fewer rows than the limit are kept intact. When
    *max_samples_per_class* is ``None`` the manifest is returned unchanged.

    When *skip_healthy* is True the healthy class is exempt from the cap
    (useful when a downstream balancing step will handle its count).
    """
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

    healthy_mask = (
        _is_healthy_mask(manifest_df, class_col=class_col) if skip_healthy else None
    )

    sampled_groups: list[pd.DataFrame] = []
    for _, group in manifest_df.groupby(class_col, dropna=False, sort=True):
        if (
            skip_healthy
            and healthy_mask is not None
            and bool(healthy_mask.loc[group.index].all())
        ):
            sampled_groups.append(group)
            continue

        if len(group) <= max_samples_per_class:
            sampled_groups.append(group)
            continue

        sampled_groups.append(
            group.sample(n=max_samples_per_class, random_state=random_seed)
        )

    if not sampled_groups:
        return manifest_df.iloc[0:0].copy()

    return pd.concat(sampled_groups, ignore_index=True)


def _is_healthy_mask(manifest_df: pd.DataFrame, class_col: str) -> pd.Series:
    """Return a boolean mask that marks healthy rows.

    Prefer ``is_healthy`` when available; otherwise infer from *class_col* by
    matching case-insensitive "healthy".
    """
    if "is_healthy" in manifest_df.columns:
        return (
            manifest_df["is_healthy"]
            .astype(str)
            .str.strip()
            .isin({"1", "True", "true"})
        )

    return manifest_df[class_col].astype(str).str.strip().str.casefold().eq("healthy")


def _balance_healthy_to_pathological(
    manifest_df: pd.DataFrame, *, random_seed: int
) -> pd.DataFrame:
    """Downsample healthy rows so their count matches the total pathological rows.

    If healthy rows already equal or are fewer than pathological rows, all
    healthy rows are kept as-is (no upsampling / duplication is ever performed
    to avoid duplicate sample_key issues in downstream merges).
    """
    if manifest_df.empty:
        return manifest_df

    if "pathology_de" in manifest_df.columns:
        class_col = "pathology_de"
    elif "pathology_en" in manifest_df.columns:
        class_col = "pathology_en"
    else:  # pragma: no cover - protected by _ensure_required_manifest_columns
        return manifest_df

    healthy_mask = _is_healthy_mask(manifest_df, class_col=class_col)
    healthy_df = manifest_df[healthy_mask].copy()
    pathological_df = manifest_df[~healthy_mask].copy()

    if healthy_df.empty or pathological_df.empty:
        return manifest_df

    target_healthy_n = len(pathological_df)
    if len(healthy_df) > target_healthy_n:
        healthy_df = healthy_df.sample(n=target_healthy_n, random_state=random_seed)

    return pd.concat([healthy_df, pathological_df], ignore_index=True)


def _to_float_mono(signal: np.ndarray, normalize: bool) -> np.ndarray:
    """Convert *signal* to a 1-D float32 array, optionally peak-normalising to [-1, 1]."""
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
    """Load a WAV file and return a status dict with the signal and metadata.

    Returns a dict with keys: ``status``, ``error``, ``signal``, ``sample_rate``,
    ``num_samples``, ``duration_seconds``. On failure ``status`` is ``"failed"``
    and ``signal`` is an empty array.
    """
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
    """Resolve a potentially relative WAV path to an absolute Path.

    Tries several candidate locations in order:
    - absolute as-is
    - relative to ``options.prefix``
    - relative to the manifest directory
    - relative to the current working directory
    - stripped of leading ``..`` components then relative to ``options.prefix``

    Returns the first candidate that exists, or the first candidate if none do.
    """
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
    """Return mean/std/min/max of *values*, ignoring non-finite entries.

    All four keys are prefixed with *prefix* (e.g. ``"ac_rms"`` → ``"ac_rms_mean"``
    etc.). Returns NaN for all stats when no finite values are present.
    """
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
    """Extract Librosa-based acoustic features from a mono float32 signal.

    Computed features (all prefixed ``ac_``):
    - Time-domain: energy, absolute mean, peak amplitude, crest factor.
    - Frame-level statistics (mean/std/min/max): RMS, ZCR, spectral centroid,
      bandwidth, roll-off, flatness.
    - MFCCs 1-13 plus their first-order deltas (mean & std per coefficient).
    - Fundamental frequency (F0) via YIN: mean/std/min/max over voiced frames.

    Returns a dict with ``acoustic_status`` (``"ok"`` or error string) and
    ``acoustic_error`` alongside the feature values.
    """
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
    """Return a logarithmically spaced array of integer DFA window scales.

    Scales range from 16 samples up to ``num_samples // 4``, with at least 6
    distinct values. Duplicate values (from int-casting) are removed.
    Returns an empty array if the signal is too short to form a valid range.
    """
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
    """Estimate the generalised Hurst exponent h(q) from MFDFA output.

    For each q-order, fits a straight line through ``log(F_q)`` vs ``log(lag)``
    and returns the slope as h(q). Requires at least 3 finite points per curve;
    q-orders that do not meet this threshold contribute NaN.

    Args:
        lags: 1-D array of window scales (length S).
        fq:   2-D fluctuation function array, shape (S, Q) or (Q, S).

    Returns:
        1-D array of h(q) values, length Q.
    """
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
    """Run MFDFA on *signal* and return summary multifractal descriptors.

    Requires the ``MFDFA`` package (``_MFDFA_AVAILABLE`` must be ``True``).

    Computed features (all prefixed ``mf_``):
    - h(q) statistics: mean, std, min, max.
    - τ(q) statistics: mean, std.
    - Singularity spectrum α: mean, std.
    - Singularity spectrum derived scalars: width (Δα), peak α, peak f(α),
      and left/right asymmetry.
    - Metadata: number of scales and q values used.

    Returns a dict with ``mf_status`` (``"ok"`` or error string) and
    ``mf_error`` alongside the feature values.
    """
    if signal.size == 0:
        return {
            "mf_status": "empty_signal",
            "mf_error": "Audio signal is empty.",
        }

    if not _MFDFA_AVAILABLE:
        return {
            "mf_status": "missing_dependency",
            "mf_error": "MFDFA package is not installed. Run: uv add MFDFA",
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

        lags, fq = _mfdfa_func(signal, lag=scales, q=q, order=options.mfdfa_order)
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


def _extract_opensmile_features(wav_path: Path, smile: Any) -> dict[str, Any]:
    """Extract eGeMAPSv02 functionals from *wav_path* using an OpenSMILE *smile* instance.

    Feature names from OpenSMILE are prefixed with ``os_`` to avoid column
    collisions with Librosa/MFDFA features.

    Returns a dict with ``opensmile_status`` (``"ok"`` or error string),
    ``opensmile_error``, and one ``os_<name>`` float entry per eGeMAPS feature.
    """
    try:
        # OpenSMILE processes the file directly
        df = smile.process_file(str(wav_path))
        if df.empty:
            return {
                "opensmile_status": "empty_result",
                "opensmile_error": "OpenSMILE returned empty dataframe.",
            }

        # The result is a dataframe with one row (since we process one file)
        # and columns are the feature names.
        features = df.iloc[0].to_dict()

        # Prefix keys to avoid collisions and ensure clean naming
        prefixed_features = {f"os_{k}": float(v) for k, v in features.items()}

        return {"opensmile_status": "ok", "opensmile_error": None, **prefixed_features}
    except Exception as exc:
        return {
            "opensmile_status": "failed",
            "opensmile_error": str(exc),
        }


def _get_smile_instance() -> Any:
    """Return a per-process cached OpenSMILE instance, or None when unavailable."""
    if not _OPENSMILE_AVAILABLE:
        return None

    global _SMILE_SINGLETON
    if _SMILE_SINGLETON is None:
        _SMILE_SINGLETON = _opensmile_mod.Smile(
            feature_set=_opensmile_mod.FeatureSet.eGeMAPSv02,
            feature_level=_opensmile_mod.FeatureLevel.Functionals,
        )
    return _SMILE_SINGLETON


def _extract_single_sample_features(
    row: dict[str, Any], options: FeatureOptions, available_meta: list[str]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Extract core/acoustic/multifractal/opensmile rows for one manifest record."""
    sample_key = str(row["sample_key"])
    wav_path_raw = row.get("wav_path")
    base_meta = {col: row.get(col) for col in available_meta}

    if not isinstance(wav_path_raw, str) or not wav_path_raw.strip():
        core_row = {
            **base_meta,
            "sample_key": sample_key,
            "feature_status": "missing_wav_path",
            "feature_error": "wav_path missing in manifest.",
            "audio_sample_rate": None,
            "audio_num_samples": 0,
            "audio_duration_seconds": 0.0,
        }
        acoustic_row = {
            "sample_key": sample_key,
            "acoustic_status": "missing_wav_path",
            "acoustic_error": "wav_path missing in manifest.",
        }
        mf_row = {
            "sample_key": sample_key,
            "mf_status": "missing_wav_path",
            "mf_error": "wav_path missing in manifest.",
        }
        os_row = {
            "sample_key": sample_key,
            "opensmile_status": "missing_wav_path",
            "opensmile_error": "wav_path missing in manifest.",
        }
        return core_row, acoustic_row, mf_row, os_row

    wav_path = _resolve_wav_path(wav_path_raw, options=options)
    if not wav_path.exists():
        core_row = {
            **base_meta,
            "sample_key": sample_key,
            "feature_status": "missing_wav_file",
            "feature_error": f"WAV file not found: {wav_path}",
            "audio_sample_rate": None,
            "audio_num_samples": 0,
            "audio_duration_seconds": 0.0,
        }
        acoustic_row = {
            "sample_key": sample_key,
            "acoustic_status": "missing_wav_file",
            "acoustic_error": f"WAV file not found: {wav_path}",
        }
        mf_row = {
            "sample_key": sample_key,
            "mf_status": "missing_wav_file",
            "mf_error": f"WAV file not found: {wav_path}",
        }
        os_row = {
            "sample_key": sample_key,
            "opensmile_status": "missing_wav_file",
            "opensmile_error": f"WAV file not found: {wav_path}",
        }
        return core_row, acoustic_row, mf_row, os_row

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
        acoustic_row = {
            "sample_key": sample_key,
            "acoustic_status": "audio_load_failed",
            "acoustic_error": audio_info["error"],
        }
        mf_row = {
            "sample_key": sample_key,
            "mf_status": "audio_load_failed",
            "mf_error": audio_info["error"],
        }
        os_row = {
            "sample_key": sample_key,
            "opensmile_status": "audio_load_failed",
            "opensmile_error": audio_info["error"],
        }
        return core_row, acoustic_row, mf_row, os_row

    signal = np.asarray(audio_info["signal"], dtype=np.float32)
    sr = int(audio_info["sample_rate"])

    acoustic_features = _extract_acoustic_features(signal, sr=sr)
    mf_features = _extract_multifractal_features(signal, options=options)

    smile = _get_smile_instance()
    if smile is not None:
        os_features = _extract_opensmile_features(wav_path, smile)
    else:
        os_features = {
            "opensmile_status": "missing_dependency",
            "opensmile_error": "opensmile package not installed.",
        }

    if (
        acoustic_features.get("acoustic_status") != "ok"
        or mf_features.get("mf_status") != "ok"
        or os_features.get("opensmile_status") != "ok"
    ):
        core_row["feature_status"] = "partial_failure"
        errors = [
            str(acoustic_features.get("acoustic_error") or ""),
            str(mf_features.get("mf_error") or ""),
            str(os_features.get("opensmile_error") or ""),
        ]
        merged_error = " | ".join([e for e in errors if e.strip()])
        core_row["feature_error"] = merged_error or None

    acoustic_row = {"sample_key": sample_key, **acoustic_features}
    mf_row = {"sample_key": sample_key, **mf_features}
    os_row = {"sample_key": sample_key, **os_features}

    return core_row, acoustic_row, mf_row, os_row


def _build_random_split_table(
    core_df: pd.DataFrame, options: FeatureOptions
) -> pd.DataFrame:
    """Build a speaker-disjoint train/val/test split table for *core_df*.

    Uses ``GroupShuffleSplit`` (grouped by ``speaker_id``) to ensure that all
    recordings from a given speaker land in exactly one split — preventing
    speaker identity leakage between train and test. Falls back to grouping
    by ``sample_key`` when ``speaker_id`` is absent.

    The returned DataFrame has columns ``sample_key``, ``split``
    (``"train"`` / ``"val"`` / ``"test"``), and ``split_seed``.
    """
    if core_df.empty or "sample_key" not in core_df.columns:
        return pd.DataFrame(columns=["sample_key", "split", "split_seed"])

    total = options.train_ratio + options.val_ratio + options.test_ratio
    if total <= 0:
        raise ValueError("train_ratio + val_ratio + test_ratio must be > 0.")

    train_ratio = options.train_ratio / total
    val_ratio = options.val_ratio / total
    test_ratio = options.test_ratio / total

    from sklearn.model_selection import GroupShuffleSplit

    # If speaker_id is missing, fallback to sample_key as group
    groups = (
        core_df["speaker_id"]
        if "speaker_id" in core_df.columns
        else core_df["sample_key"]
    )

    # First split: train_val vs test
    gss_test = GroupShuffleSplit(
        n_splits=1, test_size=test_ratio, random_state=options.random_seed
    )
    train_val_idx, test_idx = next(gss_test.split(core_df, groups=groups))

    # Second split: train vs val
    train_val_df = core_df.iloc[train_val_idx]
    train_val_groups = groups.iloc[train_val_idx]

    # Calculate val_size relative to train_val
    val_relative_ratio = (
        val_ratio / (train_ratio + val_ratio) if (train_ratio + val_ratio) > 0 else 0
    )

    if val_relative_ratio > 0:
        gss_val = GroupShuffleSplit(
            n_splits=1, test_size=val_relative_ratio, random_state=options.random_seed
        )
        train_idx_rel, val_idx_rel = next(
            gss_val.split(train_val_df, groups=train_val_groups)
        )
        train_idx = train_val_idx[train_idx_rel]
        val_idx = train_val_idx[val_idx_rel]
    else:
        train_idx = train_val_idx
        val_idx = np.array([], dtype=int)

    split_values = np.full(len(core_df), "test", dtype=object)
    split_values[train_idx] = "train"
    if len(val_idx) > 0:
        split_values[val_idx] = "val"

    return pd.DataFrame(
        {
            "sample_key": core_df["sample_key"].astype(str).tolist(),
            "split": split_values,
            "split_seed": options.random_seed,
        }
    )


def _prepare_target_manifest(options: FeatureOptions) -> pd.DataFrame:
    """Load the dataset manifest and apply per-class sampling limits.

    Builds the manifest from raw data if it is missing, then downsamples
    each class to ``options.max_samples_per_class`` rows (when set).
    """
    base_pipeline_options = PipelineOptions(prefix=options.prefix)
    manifest_df = load_dataset_dataframe(
        manifest_path=options.input_manifest,
        build_if_missing=True,
        options=base_pipeline_options,
        save_if_built=True,
    )
    _ensure_required_manifest_columns(manifest_df)

    # Filter to a single token before sampling (avoids loading all tokens)
    if options.selected_token is not None and "token" in manifest_df.columns:
        manifest_df = manifest_df[manifest_df["token"] == options.selected_token].copy()

    manifest_df = _limit_samples_per_class(
        manifest_df,
        max_samples_per_class=options.max_samples_per_class,
        random_seed=options.random_seed,
        skip_healthy=options.balance_healthy,
    )

    if options.balance_healthy:
        manifest_df = _balance_healthy_to_pathological(
            manifest_df,
            random_seed=options.random_seed,
        )

    return manifest_df


def _extract_feature_tables_from_manifest(
    manifest_df: pd.DataFrame, options: FeatureOptions
) -> dict[str, pd.DataFrame]:
    """Extract all feature tables for every row in *manifest_df*.

    For each sample the pipeline:
    1. Resolves and loads the WAV file.
    2. Extracts Librosa acoustic features.
    3. Extracts MFDFA multifractal features (skipped if MFDFA unavailable).
    4. Extracts OpenSMILE eGeMAPSv02 features (skipped if opensmile unavailable).
    5. Builds speaker-disjoint splits (when ``options.include_splits`` is True).

    Returns a dict with keys ``"core"``, ``"acoustic"``, ``"multifractal"``,
    ``"opensmile"``, and optionally ``"splits"``.
    """
    if manifest_df.empty:
        tables: dict[str, pd.DataFrame] = {
            "core": pd.DataFrame(),
            "acoustic": pd.DataFrame(),
            "multifractal": pd.DataFrame(),
            "opensmile": pd.DataFrame(),
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
    opensmile_rows: list[dict[str, Any]] = []

    records = [
        {str(k): v for k, v in row.items()}
        for row in manifest_df.to_dict(orient="records")
    ]
    requested_workers = options.num_workers
    if requested_workers is None or requested_workers <= 0:
        requested_workers = max((os.cpu_count() or 2) - 1, 1)
    requested_workers = min(requested_workers, max(len(records), 1))

    if requested_workers > 1 and len(records) > 1:
        results: list[
            tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]] | None
        ] = [None] * len(records)

        with ProcessPoolExecutor(max_workers=requested_workers) as executor:
            futures = {
                executor.submit(
                    _extract_single_sample_features, row, options, available_meta
                ): idx
                for idx, row in enumerate(records)
            }

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Extracting features",
                unit="sample",
            ):
                idx = futures[future]
                results[idx] = future.result()

        for result in results:
            if result is None:  # pragma: no cover - defensive path
                continue
            core_row, acoustic_row, mf_row, os_row = result
            core_rows.append(core_row)
            acoustic_rows.append(acoustic_row)
            multifractal_rows.append(mf_row)
            opensmile_rows.append(os_row)
    else:
        for row in tqdm(records, desc="Extracting features", unit="sample"):
            core_row, acoustic_row, mf_row, os_row = _extract_single_sample_features(
                row, options, available_meta
            )
            core_rows.append(core_row)
            acoustic_rows.append(acoustic_row)
            multifractal_rows.append(mf_row)
            opensmile_rows.append(os_row)

    core_df = pd.DataFrame(core_rows)
    acoustic_df = pd.DataFrame(acoustic_rows)
    multifractal_df = pd.DataFrame(multifractal_rows)
    opensmile_df = pd.DataFrame(opensmile_rows)

    tables: dict[str, pd.DataFrame] = {
        "core": core_df,
        "acoustic": acoustic_df,
        "multifractal": multifractal_df,
        "opensmile": opensmile_df,
    }

    if options.include_splits:
        split_df = _build_random_split_table(
            core_df=core_df,
            options=options,
        )
        tables["splits"] = split_df

    return tables


def extract_feature_tables(options: FeatureOptions) -> dict[str, pd.DataFrame]:
    """Build all feature tables from scratch using *options*.

    This is the top-level entry point for a full extraction run. It loads
    (or builds) the manifest, applies sampling limits, and runs the complete
    feature extraction pipeline.

    Prefer ``load_feature_tables`` from ``feature_cache`` for incremental /
    cache-aware loading.
    """
    manifest_df = _prepare_target_manifest(options)
    return _extract_feature_tables_from_manifest(manifest_df, options)
