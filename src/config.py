"""Project configuration constants for dataset preprocessing and loading."""

from __future__ import annotations

from pathlib import Path

# Root data directories.
RAW_DATA_ROOT = Path("data/raw")
PROCESSED_DATA_ROOT = Path("data/processed")
DEFAULT_WAV_ROOT = PROCESSED_DATA_ROOT / "wav"
DEFAULT_MANIFEST_PATH = PROCESSED_DATA_ROOT / "manifests" / "dataset_manifest.csv"

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
    "Reinke Ödem": "reinke's edema",
    "Rekurrensparese": "recurrent laryngeal nerve paralysis",
    "Spasmodische Dysphonie": "spasmodic dysphonia",
    "Stimmlippenpolyp": "vocal fold polyp",
}

ENCODING_CANDIDATES = ("utf-8", "utf-8-sig", "cp1252", "latin-1")
