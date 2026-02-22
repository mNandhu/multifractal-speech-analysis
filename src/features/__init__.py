"""Feature engineering and extraction APIs."""

from src.features.feature_cache import (
    load_feature_tables,
    save_feature_summary_json,
    save_feature_tables,
    summarize_feature_tables,
)
from src.features.feature_extraction import extract_feature_tables
from src.features.feature_options import FeatureOptions


def build_feature_tables(options: FeatureOptions):
    """Backward-compatible alias for :func:`extract_feature_tables`."""
    return extract_feature_tables(options)


__all__ = [
    "FeatureOptions",
    "build_feature_tables",
    "extract_feature_tables",
    "load_feature_tables",
    "save_feature_summary_json",
    "save_feature_tables",
    "summarize_feature_tables",
]
