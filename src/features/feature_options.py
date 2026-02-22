from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.config import (
    DEFAULT_FEATURE_ACOUSTIC_PATH,
    DEFAULT_FEATURE_CORE_PATH,
    DEFAULT_FEATURE_MULTIFRACTAL_PATH,
    DEFAULT_FEATURE_SPLITS_PATH,
    DEFAULT_FEATURE_SUMMARY_JSON_PATH,
    DEFAULT_MANIFEST_PATH,
)


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
