from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class DetectorConfig:
    temporal_sample_interval_sec: float = 2.0
    temporal_confirmation_sec: float = 10.0
    temporal_changed_block_ratio: float = 0.50
    temporal_same_content_similarity: float = 0.80
    temporal_analysis_width: int = 320
    temporal_analysis_height: int = 180
    temporal_chunk_duration_sec: float = 300.0
    auto_detect_screen_crop: bool = True
    screen_crop_left_ratio: float = 0.10
    screen_crop_top_ratio: float = 0.02
    screen_crop_right_ratio: float = 0.10
    screen_crop_bottom_ratio: float = 0.10
    crop_output_screenshots: bool = True
    grid_columns: int = 4
    grid_rows: int = 3
    block_hash_distance: int = 10
    temporal_refinement_fps: float = 4.0
    min_page_duration_sec: float = 5.0
    jpeg_quality: int = 90
    ffmpeg_path: str | None = None
    ffprobe_path: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "DetectorConfig":
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"Unknown configuration keys: {', '.join(unknown)}")
        config = cls(**dict(data))
        config.validate()
        return config

    @classmethod
    def from_file(cls, path: str | Path) -> "DetectorConfig":
        config_path = Path(path)
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON configuration: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("Configuration root must be a JSON object")
        return cls.from_mapping(data)

    def validate(self) -> None:
        positive = {
            "temporal_sample_interval_sec": self.temporal_sample_interval_sec,
            "temporal_confirmation_sec": self.temporal_confirmation_sec,
            "temporal_analysis_width": self.temporal_analysis_width,
            "temporal_analysis_height": self.temporal_analysis_height,
            "temporal_chunk_duration_sec": self.temporal_chunk_duration_sec,
            "temporal_refinement_fps": self.temporal_refinement_fps,
            "min_page_duration_sec": self.min_page_duration_sec,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"Values must be positive: {', '.join(invalid)}")
        if self.temporal_confirmation_sec < self.temporal_sample_interval_sec * 2:
            raise ValueError(
                "temporal_confirmation_sec must cover at least two samples"
            )
        if not 0.0 < self.temporal_changed_block_ratio <= 1.0:
            raise ValueError(
                "temporal_changed_block_ratio must be in the range (0, 1]"
            )
        if not 0.0 < self.temporal_same_content_similarity <= 1.0:
            raise ValueError(
                "temporal_same_content_similarity must be in the range (0, 1]"
            )
        crop_values = {
            "screen_crop_left_ratio": self.screen_crop_left_ratio,
            "screen_crop_top_ratio": self.screen_crop_top_ratio,
            "screen_crop_right_ratio": self.screen_crop_right_ratio,
            "screen_crop_bottom_ratio": self.screen_crop_bottom_ratio,
        }
        invalid_crops = [
            name for name, value in crop_values.items() if not 0.0 <= value < 1.0
        ]
        if invalid_crops:
            raise ValueError(
                f"screen crop ratios must be in [0, 1): {', '.join(invalid_crops)}"
            )
        if self.screen_crop_left_ratio + self.screen_crop_right_ratio >= 1.0:
            raise ValueError("horizontal screen crop ratios must sum to less than 1")
        if self.screen_crop_top_ratio + self.screen_crop_bottom_ratio >= 1.0:
            raise ValueError("vertical screen crop ratios must sum to less than 1")
        if self.grid_columns < 1 or self.grid_rows < 1:
            raise ValueError("grid dimensions must be positive")
        if not 0 <= self.block_hash_distance <= 64:
            raise ValueError("block_hash_distance must be between 0 and 64")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
