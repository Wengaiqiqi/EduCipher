"""Data models used only by the legacy scene detector."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class WorkingPage:
    start_sec: float
    frame: np.ndarray
    signature: np.ndarray
    confidence: str
    note: str | None = None
    representative_sec: float = 0.0
    analysis_frame: np.ndarray | None = None


@dataclass(frozen=True)
class CandidateAudit:
    candidate_id: int
    timestamp_sec: float
    change_ratio: float
    mean_hamming_distance: float
    block_distances: list[int]
    classification: str
    stable: bool | None
    stable_frame_sec: float | None
    stable_difference: float | None
    decision: str
    duplicate_hash_distance: int | None = None
    duplicate_changed_block_ratio: float | None = None
    files: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "timestamp_sec": round(self.timestamp_sec, 3),
            "change_ratio": round(self.change_ratio, 4),
            "mean_hamming_distance": round(self.mean_hamming_distance, 3),
            "block_distances": self.block_distances,
            "classification": self.classification,
            "stable": self.stable,
            "stable_frame_sec": (
                round(self.stable_frame_sec, 3)
                if self.stable_frame_sec is not None
                else None
            ),
            "stable_difference": (
                round(self.stable_difference, 5)
                if self.stable_difference is not None
                else None
            ),
            "decision": self.decision,
            "duplicate_hash_distance": self.duplicate_hash_distance,
            "duplicate_changed_block_ratio": (
                round(self.duplicate_changed_block_ratio, 4)
                if self.duplicate_changed_block_ratio is not None
                else None
            ),
            "files": self.files,
        }
