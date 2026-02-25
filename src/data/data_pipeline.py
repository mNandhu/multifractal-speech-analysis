from __future__ import annotations

import json
import unicodedata
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import cast

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import nspfile
from src.config import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_WAV_ROOT,
    ENCODING_CANDIDATES,
    GERMAN_TO_ENGLISH_COLUMNS,
    PATHOLOGY_DE_TO_EN,
    RAW_DATA_ROOT,
)


@dataclass(slots=True)
class PipelineOptions:
    prefix: Path | str = Path(".")
    data_root: Path = RAW_DATA_ROOT
    wav_root: Path = DEFAULT_WAV_ROOT
    output_manifest: Path = DEFAULT_MANIFEST_PATH
    export_csv: bool = True
    overwrite_wav: bool = False

    @property
    def prefix_path(self) -> Path:
        return Path(self.prefix)

    def resolve_path(self, path: Path | str) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        return self.prefix_path / candidate

    @property
    def resolved_data_root(self) -> Path:
        return self.resolve_path(self.data_root)

    @property
    def resolved_wav_root(self) -> Path:
        return self.resolve_path(self.wav_root)

    @property
    def resolved_output_manifest(self) -> Path:
        return self.resolve_path(self.output_manifest)


# Backwards-compatible alias.
BuildOptions = PipelineOptions


def _normalize_text(value: str | None) -> str | None:
    if value is None:
        return None
    return unicodedata.normalize("NFC", str(value)).strip()


def _safe_read_csv(csv_path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ENCODING_CANDIDATES:
        try:
            return pd.read_csv(csv_path, encoding=encoding)
        except Exception as exc:  # pragma: no cover - defensive path
            last_error = exc
    raise RuntimeError(f"Could not read CSV: {csv_path}") from last_error


def _load_overview(pathology_dir: Path) -> pd.DataFrame:
    overview_path = pathology_dir / "overview.csv"
    if not overview_path.exists():
        return pd.DataFrame()

    df = _safe_read_csv(overview_path)
    df.columns = [_normalize_text(col) for col in df.columns]

    missing = [col for col in GERMAN_TO_ENGLISH_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing expected columns in {overview_path}: {missing}. "
            f"Found columns={list(df.columns)}"
        )

    renamed = df.rename(columns=GERMAN_TO_ENGLISH_COLUMNS)
    renamed["recording_id"] = renamed["recording_id"].astype(str).str.strip()
    renamed["speaker_id"] = renamed["speaker_id"].astype(str).str.strip()
    renamed["sex"] = renamed["sex"].astype(str).str.strip().str.lower()
    renamed["recording_date"] = pd.to_datetime(
        renamed["recording_date"], errors="coerce"
    )
    renamed["birth_date"] = pd.to_datetime(renamed["birth_date"], errors="coerce")
    renamed["pathology_csv_raw"] = renamed["pathologies_de"].astype(str).str.strip()
    renamed["overview_path"] = str(overview_path)

    return renamed


def _to_int16_audio(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data)

    if arr.ndim == 1:
        arr = arr[:, np.newaxis]

    if arr.dtype == np.int16:
        return arr

    if np.issubdtype(arr.dtype, np.integer):
        clipped = np.clip(arr, -32768, 32767)
        return clipped.astype(np.int16)

    # Float path: normalize into int16 range.
    finite = arr[np.isfinite(arr)]
    max_abs = float(np.max(np.abs(finite))) if finite.size else 0.0
    if max_abs == 0.0:
        return np.zeros(arr.shape, dtype=np.int16)

    scaled = np.clip(arr / max_abs, -1.0, 1.0)
    return (scaled * 32767.0).astype(np.int16)


def convert_nsp_to_wav(
    nsp_path: Path, wav_path: Path, overwrite: bool = False
) -> dict[str, Any]:
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    if wav_path.exists() and not overwrite:
        return {
            "wav_conversion_status": "skipped_exists",
            "wav_sample_rate": None,
            "wav_num_frames": None,
            "wav_num_channels": None,
            "wav_error": None,
        }

    try:
        read_result = nspfile.read(str(nsp_path))
        if not isinstance(read_result, tuple) or len(read_result) < 2:
            raise ValueError(f"Unexpected NSP read output for {nsp_path}")

        sample_rate = int(read_result[0])
        data = cast(np.ndarray, read_result[1])
        audio = _to_int16_audio(data)

        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(audio.shape[1])
            wf.setsampwidth(2)  # int16
            wf.setframerate(int(sample_rate))
            wf.writeframes(audio.tobytes(order="C"))

        return {
            "wav_conversion_status": "converted",
            "wav_sample_rate": int(sample_rate),
            "wav_num_frames": int(audio.shape[0]),
            "wav_num_channels": int(audio.shape[1]),
            "wav_error": None,
        }
    except Exception as exc:  # pragma: no cover - defensive path
        return {
            "wav_conversion_status": "failed",
            "wav_sample_rate": None,
            "wav_num_frames": None,
            "wav_num_channels": None,
            "wav_error": str(exc),
        }


def _parse_sample_token(filename_stem: str, recording_id: str) -> str:
    prefix = f"{recording_id}-"
    if filename_stem.startswith(prefix):
        return filename_stem[len(prefix) :]
    return filename_stem


def _pathology_to_english(pathology_de: str) -> str:
    normalized = _normalize_text(pathology_de) or pathology_de
    return PATHOLOGY_DE_TO_EN.get(normalized, normalized)


def preprocess_dataset(options: PipelineOptions) -> pd.DataFrame:
    return _preprocess_dataset_internal(options=options, include_pathologies=None)


def _preprocess_dataset_internal(
    *, options: PipelineOptions, include_pathologies: set[str] | None
) -> pd.DataFrame:
    data_root = options.resolved_data_root
    wav_root = options.resolved_wav_root
    rows: list[dict[str, Any]] = []

    pathology_dirs = sorted(
        [p for p in data_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower()
    )

    if include_pathologies is not None:
        pathology_dirs = [p for p in pathology_dirs if p.name in include_pathologies]

    for pathology_dir in tqdm(
        pathology_dirs, desc="Pathologies", unit="pathology", position=0
    ):
        pathology_de = pathology_dir.name
        pathology_en = _pathology_to_english(pathology_de)

        overview_df = _load_overview(pathology_dir)
        overview_by_recording: dict[str, pd.Series] = {}

        if not overview_df.empty:
            for rid, group in overview_df.groupby("recording_id", dropna=False):
                overview_by_recording[str(rid)] = group.iloc[0]

        recording_dirs = sorted(
            [p for p in pathology_dir.iterdir() if p.is_dir() and p.name.isdigit()],
            key=lambda p: int(p.name),
        )

        for recording_dir in tqdm(
            recording_dirs,
            desc=f"{pathology_de} recordings",
            unit="recording",
            leave=False,
            position=1,
        ):
            recording_id = recording_dir.name
            remarks_path = recording_dir / "remarks" / f"{recording_id}-remarks.txt"

            overview_row = overview_by_recording.get(recording_id)
            overview_match_status = "matched" if overview_row is not None else "missing"

            base_meta = {
                "recording_id": recording_id,
                "pathology_de": pathology_de,
                "pathology_en": pathology_en,
                "duplicate_class_key": f"{pathology_de}::{recording_id}",
                "recording_dir": str(recording_dir),
                "remarks_path": str(remarks_path) if remarks_path.exists() else None,
                "has_remarks": bool(remarks_path.exists()),
                "overview_match_status": overview_match_status,
            }

            if overview_row is not None:
                base_meta.update(
                    {
                        "recording_type": overview_row.get("recording_type"),
                        "recording_date": overview_row.get("recording_date"),
                        "diagnosis_de": overview_row.get("diagnosis_de"),
                        "speaker_id": str(overview_row.get("speaker_id"))
                        if pd.notna(overview_row.get("speaker_id"))
                        else None,
                        "birth_date": overview_row.get("birth_date"),
                        "sex": overview_row.get("sex"),
                        "pathology_csv_raw": overview_row.get("pathology_csv_raw"),
                        "overview_path": overview_row.get("overview_path"),
                    }
                )
            else:
                base_meta.update(
                    {
                        "recording_type": None,
                        "recording_date": None,
                        "diagnosis_de": None,
                        "speaker_id": None,
                        "birth_date": None,
                        "sex": None,
                        "pathology_csv_raw": None,
                        "overview_path": str(pathology_dir / "overview.csv"),
                    }
                )

            nsp_files: list[tuple[str, Path]] = []
            for modality in ("vowels", "sentences"):
                modality_dir = recording_dir / modality
                if not modality_dir.exists():
                    continue

                for nsp_path in sorted(modality_dir.glob("*.nsp")):
                    nsp_files.append((modality, nsp_path))

            for modality, nsp_path in tqdm(
                nsp_files,
                desc=f"{pathology_de}/{recording_id} files",
                unit="file",
                leave=False,
                position=2,
            ):
                stem = nsp_path.stem
                token = _parse_sample_token(stem, recording_id=recording_id)

                modality_dir = nsp_path.parent
                egg_name = f"{stem}-egg.egg"
                egg_path = modality_dir / egg_name

                rel_parent = nsp_path.parent.relative_to(data_root)
                wav_path = wav_root / rel_parent / f"{stem}.wav"
                conversion = convert_nsp_to_wav(
                    nsp_path=nsp_path,
                    wav_path=wav_path,
                    overwrite=options.overwrite_wav,
                )

                row = {
                    **base_meta,
                    "sample_key": f"{pathology_de}::{recording_id}::{modality}::{token}",
                    "modality": modality,
                    "token": token,
                    "nsp_path": str(nsp_path),
                    "wav_path": str(wav_path),
                    "egg_path": str(egg_path) if egg_path.exists() else None,
                    "has_egg": bool(egg_path.exists()),
                    **conversion,
                }
                rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # Normalize key fields for downstream use.
    df["is_healthy"] = df["pathology_de"].str.lower().eq("healthy")
    recording_series = df["recording_id"].fillna("").astype(str).str.strip()
    recording_overlap_counts = (
        df.loc[recording_series.ne(""), ["recording_id", "pathology_de"]]
        .drop_duplicates()
        .groupby("recording_id", dropna=False)["pathology_de"]
        .nunique()
    )
    overlap_recording_set = set(
        recording_overlap_counts[recording_overlap_counts > 1].index.astype(str)
    )
    df["is_overlap_speaker"] = recording_series.isin(overlap_recording_set)

    speaker_series = df["speaker_id"].fillna("").astype(str).str.strip()
    speaker_overlap_counts = (
        df.loc[speaker_series.ne(""), ["speaker_id", "pathology_de"]]
        .drop_duplicates()
        .groupby("speaker_id", dropna=False)["pathology_de"]
        .nunique()
    )
    overlap_speaker_id_set = set(
        speaker_overlap_counts[speaker_overlap_counts > 1].index.astype(str)
    )
    df["is_overlap_speaker_id"] = speaker_series.isin(overlap_speaker_id_set)

    ordered_columns = [
        "sample_key",
        "duplicate_class_key",
        "recording_id",
        "speaker_id",
        "is_overlap_speaker",
        "is_overlap_speaker_id",
        "recording_type",
        "recording_date",
        "birth_date",
        "sex",
        "diagnosis_de",
        "pathology_de",
        "pathology_en",
        "pathology_csv_raw",
        "is_healthy",
        "modality",
        "token",
        "nsp_path",
        "wav_path",
        "egg_path",
        "has_egg",
        "remarks_path",
        "has_remarks",
        "recording_dir",
        "overview_path",
        "overview_match_status",
        "wav_conversion_status",
        "wav_sample_rate",
        "wav_num_frames",
        "wav_num_channels",
        "wav_error",
    ]

    available_ordered = [col for col in ordered_columns if col in df.columns]
    remainder = [col for col in df.columns if col not in available_ordered]

    return df[available_ordered + remainder]


def _raw_pathology_names(options: PipelineOptions) -> set[str]:
    data_root = options.resolved_data_root
    if not data_root.exists():
        return set()
    return {p.name for p in data_root.iterdir() if p.is_dir()}


def _manifest_pathology_names(df: pd.DataFrame) -> set[str]:
    if df.empty or "pathology_de" not in df.columns:
        return set()
    series = df["pathology_de"].astype(str).str.strip()
    return set(series[series.ne("")].unique())


def _append_new_raw_classes_to_manifest(
    existing_df: pd.DataFrame, *, options: PipelineOptions
) -> tuple[pd.DataFrame, list[str]]:
    """Process only new pathology folders and append rows to existing manifest.

    Returns:
        (updated_df, processed_class_names)
    """
    raw_classes = _raw_pathology_names(options)
    manifest_classes = _manifest_pathology_names(existing_df)
    missing_classes = sorted(raw_classes - manifest_classes)

    if not missing_classes:
        return existing_df, []

    new_df = _preprocess_dataset_internal(
        options=options, include_pathologies=set(missing_classes)
    )

    if new_df.empty:
        return existing_df, []

    merged = pd.concat([existing_df, new_df], ignore_index=True)
    if "sample_key" in merged.columns:
        merged = merged.drop_duplicates(subset=["sample_key"], keep="last")

    return merged, missing_classes


def build_unified_dataframe(options: PipelineOptions) -> pd.DataFrame:
    """Backward-compatible alias for preprocess_dataset."""
    return preprocess_dataset(options)


def save_dataset_dataframe(
    df: pd.DataFrame, output_manifest: Path, export_csv: bool = True
) -> None:
    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    if output_manifest.suffix.lower() == ".parquet":
        df.to_parquet(output_manifest, index=False)
    elif output_manifest.suffix.lower() == ".csv":
        df.to_csv(output_manifest, index=False)
    else:
        raise ValueError("output_manifest must end with .parquet or .csv")

    if export_csv and output_manifest.suffix.lower() != ".csv":
        csv_out = output_manifest.with_suffix(".csv")
        df.to_csv(csv_out, index=False)


def save_unified_dataframe(
    df: pd.DataFrame, output_manifest: Path, export_csv: bool = True
) -> None:
    """Backward-compatible alias for save_dataset_dataframe."""
    save_dataset_dataframe(df, output_manifest=output_manifest, export_csv=export_csv)


def load_dataset_dataframe(
    manifest_path: Path | None = None,
    *,
    build_if_missing: bool = True,
    options: PipelineOptions | None = None,
    save_if_built: bool = True,
    append_new_raw_classes: bool = True,
) -> pd.DataFrame:
    """
    Load the preprocessed dataset dataframe on demand.

    If the manifest does not exist and ``build_if_missing=True``, the pipeline
    preprocesses data (including NSP->WAV conversion), optionally saves outputs,
    and returns the dataframe.
    """
    effective_options = options or PipelineOptions()
    if manifest_path is not None:
        target_manifest = effective_options.resolve_path(manifest_path)
    else:
        target_manifest = effective_options.resolved_output_manifest

    if target_manifest.exists():
        suffix = target_manifest.suffix.lower()
        if suffix == ".csv":
            existing_df = pd.read_csv(target_manifest)
        elif suffix == ".parquet":
            existing_df = pd.read_parquet(target_manifest)
        else:
            raise ValueError("manifest_path must end with .csv or .parquet")

        if build_if_missing and append_new_raw_classes:
            updated_df, added_classes = _append_new_raw_classes_to_manifest(
                existing_df, options=effective_options
            )
            if added_classes and save_if_built:
                save_dataset_dataframe(
                    updated_df,
                    output_manifest=target_manifest,
                    export_csv=effective_options.export_csv,
                )
            return updated_df

        return existing_df

    if not build_if_missing:
        raise FileNotFoundError(
            f"Manifest not found: {target_manifest}. "
            "Set build_if_missing=True to preprocess and build it automatically."
        )

    df = preprocess_dataset(effective_options)
    if save_if_built:
        save_dataset_dataframe(
            df,
            output_manifest=target_manifest,
            export_csv=effective_options.export_csv,
        )
    return df


def summarize_manifest(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "num_rows": 0,
            "num_pathologies": 0,
            "num_recordings": 0,
            "num_speakers": 0,
            "wav_status": {},
        }

    wav_status = df["wav_conversion_status"].value_counts(dropna=False).to_dict()
    return {
        "num_rows": int(len(df)),
        "num_pathologies": int(df["pathology_de"].nunique(dropna=True)),
        "num_recordings": int(df["duplicate_class_key"].nunique(dropna=True)),
        "num_speakers": int(df["speaker_id"].nunique(dropna=True)),
        "wav_status": wav_status,
    }


def save_summary_json(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
