"""Project configuration constants for dataset preprocessing and loading."""

from __future__ import annotations

from pathlib import Path

# Root data directories.
RAW_DATA_ROOT = Path("data/raw")
PROCESSED_DATA_ROOT = Path("data/processed")
DEFAULT_WAV_ROOT = PROCESSED_DATA_ROOT / "wav"
DEFAULT_MANIFEST_PATH = PROCESSED_DATA_ROOT / "manifests" / "dataset_manifest.csv"
DEFAULT_FEATURES_ROOT = PROCESSED_DATA_ROOT / "features"
DEFAULT_FEATURE_CORE_PATH = DEFAULT_FEATURES_ROOT / "sample_core.csv"
DEFAULT_FEATURE_ACOUSTIC_PATH = DEFAULT_FEATURES_ROOT / "acoustic_features.csv"
DEFAULT_FEATURE_MULTIFRACTAL_PATH = DEFAULT_FEATURES_ROOT / "multifractal_features.csv"
DEFAULT_FEATURE_OPENSMILE_PATH = DEFAULT_FEATURES_ROOT / "opensmile_features.csv"
DEFAULT_FEATURE_NEUROKIT2_PATH = DEFAULT_FEATURES_ROOT / "neurokit2_features.csv"
DEFAULT_FEATURE_SPLITS_PATH = DEFAULT_FEATURES_ROOT / "sample_splits.csv"
DEFAULT_FEATURE_SUMMARY_JSON_PATH = DEFAULT_FEATURES_ROOT / "feature_summary.json"

# German -> English metadata column mapping.
GERMAN_TO_ENGLISH_COLUMNS = {
    "AufnahmeID": "recording_id",
    "AufnahmeTyp": "recording_type",
    "AufnahmeDatum": "recording_date",
    "Diagnose": "diagnosis_de",
    "SprecherID": "speaker_id",
    "Geburtsdatum": "birth_date",
    "Geschlecht": "sex",
    "Pathologien": "pathologies_de",
}

# Canonical German pathology folder names -> English labels.
PATHOLOGY_DE_TO_EN = {
    "healthy": "healthy",
    "Morbus Parkinson": "parkinson's disease",
    "Phonationsknötchen": "vocal fold nodules",
    "Phonationsknötchen": "vocal fold nodules",  # Handle both "ö" and "ö" variants.
    "Reinke Ödem": "reinke's edema",
    "Reinke Ödem": "reinke's edema",  # Handle both "ö" and "ö" variants.
    "Rekurrensparese": "recurrent laryngeal nerve paralysis",
    "Spasmodische Dysphonie": "spasmodic dysphonia",
    "Stimmlippenpolyp": "vocal fold polyp",
    "Hyperfunktionelle Dysphonie": "hyperfunctional dysphonia",
    "Funktionelle Dysphonie": "functional dysphonia",
    "Psychogene Dysphonie": "psychogenic dysphonia",
    "Hypotone Dysphonie": "hypotonic dysphonia",
    "Laryngitis": "laryngitis",
}

ENCODING_CANDIDATES = ("utf-8", "utf-8-sig", "cp1252", "latin-1")
