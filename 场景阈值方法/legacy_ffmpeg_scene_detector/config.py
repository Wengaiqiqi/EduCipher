from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class DetectorConfig:
    scene_threshold: float = 0.05
    adaptive_scene_detection: bool = True
    scene_threshold_floor: float = 0.05
    minimum_scene_candidates_per_minute: float = 0.5
    scene_scan_fps: float = 2.0
    auto_detect_screen_crop: bool = True
    screen_crop_left_ratio: float = 0.10
    screen_crop_top_ratio: float = 0.02
    screen_crop_right_ratio: float = 0.10
    screen_crop_bottom_ratio: float = 0.10
    crop_output_screenshots: bool = True
    grid_columns: int = 4
    grid_rows: int = 3
    block_hash_distance: int = 10
    high_change_ratio: float = 0.65
    low_change_ratio: float = 0.25
    comparison_offset_sec: float = 0.5
    stable_delay_sec: float = 0.5
    stable_window_sec: float = 3.0
    stable_sample_fps: float = 4.0
    stable_diff_threshold: float = 0.025
    min_page_duration_sec: float = 5.0
    duplicate_hash_distance: int = 6
    duplicate_changed_block_ratio: float = 0.25
    no_ppt_min_duration_sec: float = 30.0
    no_ppt_merge_gap_sec: float = 5.0
    analysis_width: int = 640
    analysis_height: int = 360
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
        if not 0.0 <= self.scene_threshold <= 1.0:
            raise ValueError("scene_threshold must be between 0 and 1")
        if not 0.0 < self.scene_threshold_floor <= self.scene_threshold:
            raise ValueError(
                "scene_threshold_floor must be positive and no greater than "
                "scene_threshold"
            )
        if self.minimum_scene_candidates_per_minute < 0:
            raise ValueError(
                "minimum_scene_candidates_per_minute cannot be negative"
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
        if not 0.0 <= self.low_change_ratio < self.high_change_ratio <= 1.0:
            raise ValueError(
                "change thresholds must satisfy 0 <= low < high <= 1"
            )
        positive = {
            "scene_scan_fps": self.scene_scan_fps,
            "comparison_offset_sec": self.comparison_offset_sec,
            "stable_window_sec": self.stable_window_sec,
            "stable_sample_fps": self.stable_sample_fps,
            "min_page_duration_sec": self.min_page_duration_sec,
            "no_ppt_min_duration_sec": self.no_ppt_min_duration_sec,
            "analysis_width": self.analysis_width,
            "analysis_height": self.analysis_height,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"Values must be positive: {', '.join(invalid)}")
        if self.stable_delay_sec < 0 or self.no_ppt_merge_gap_sec < 0:
            raise ValueError("delay and merge gap cannot be negative")
        if not 0.0 <= self.stable_diff_threshold <= 1.0:
            raise ValueError("stable_diff_threshold must be between 0 and 1")
        if not 0 <= self.duplicate_hash_distance <= 64:
            raise ValueError("duplicate_hash_distance must be between 0 and 64")
        if not 0.0 <= self.duplicate_changed_block_ratio <= 1.0:
            raise ValueError(
                "duplicate_changed_block_ratio must be between 0 and 1"
            )
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
