"""Image analysis helpers for the archived scene detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from .hashing import frame_difference

Confidence = Literal["high", "low"]


def detect_screen_crop_ratios(
    frames: Sequence[np.ndarray],
    *,
    fallback: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    if not frames:
        return fallback
    shape = frames[0].shape
    if any(frame.shape != shape for frame in frames):
        raise ValueError("calibration frames must have identical shapes")
    median_frame = np.median(np.stack(frames), axis=0)
    gray = (
        median_frame.mean(axis=2)
        if median_frame.ndim == 3
        else median_frame
    )
    height, width = gray.shape
    vertical_edges = np.abs(np.diff(gray, axis=1)).mean(axis=0)
    horizontal_edges = np.abs(np.diff(gray, axis=0)).mean(axis=1)
    kernel = np.ones(5, dtype=np.float64) / 5.0
    vertical_edges = np.convolve(vertical_edges, kernel, mode="same")
    horizontal_edges = np.convolve(horizontal_edges, kernel, mode="same")

    left_search = slice(max(1, int(width * 0.02)), int(width * 0.35))
    right_search = slice(int(width * 0.65), max(1, int(width * 0.98)))
    top_search = slice(0, max(1, int(height * 0.25)))
    bottom_search = slice(int(height * 0.65), max(1, int(height * 0.98)))
    left = left_search.start + int(np.argmax(vertical_edges[left_search]))
    right = right_search.start + int(np.argmax(vertical_edges[right_search]))
    top = top_search.start + int(np.argmax(horizontal_edges[top_search]))
    bottom = bottom_search.start + int(np.argmax(horizontal_edges[bottom_search]))

    left = max(0, left - int(width * 0.10))
    right = min(width, right + int(width * 0.10))
    top = max(0, top - int(height * 0.08))
    bottom = min(height, bottom + int(height * 0.15))
    if right - left < width * 0.55 or bottom - top < height * 0.55:
        return fallback

    detected = (
        max(0.02, left / width),
        max(0.0, top / height),
        max(0.02, 1.0 - right / width),
        max(0.05, 1.0 - bottom / height),
    )
    if detected[0] + detected[2] >= 0.45:
        return fallback
    if detected[1] + detected[3] >= 0.40:
        return fallback
    return detected


def crop_frame(
    frame: np.ndarray,
    *,
    left_ratio: float,
    top_ratio: float,
    right_ratio: float,
    bottom_ratio: float,
) -> np.ndarray:
    if frame.ndim not in (2, 3):
        raise ValueError("frame must be a grayscale or RGB array")
    height, width = frame.shape[:2]
    x0 = int(round(width * left_ratio))
    x1 = int(round(width * (1.0 - right_ratio)))
    y0 = int(round(height * top_ratio))
    y1 = int(round(height * (1.0 - bottom_ratio)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("screen crop produces an empty frame")
    return frame[y0:y1, x0:x1]


def classify_change(
    change_ratio: float,
    *,
    low_threshold: float,
    high_threshold: float,
) -> Confidence | None:
    if change_ratio >= high_threshold:
        return "high"
    if change_ratio > low_threshold:
        return "low"
    return None


@dataclass(frozen=True)
class StableSelection:
    index: int
    difference: float
    stable: bool


def select_stable_frame(
    frames: Sequence[np.ndarray],
    *,
    first_index: int,
    threshold: float,
) -> StableSelection:
    """Choose the first stable frame or the least-changing fallback frame."""
    if not frames:
        raise ValueError("at least one frame is required")
    start = min(max(first_index, 1), len(frames) - 1)
    best_index = start
    best_difference = float("inf")
    for index in range(start, len(frames)):
        difference = frame_difference(frames[index - 1], frames[index])
        if difference < best_difference:
            best_index = index
            best_difference = difference
        if difference <= threshold:
            return StableSelection(index=index, difference=difference, stable=True)
    if best_difference == float("inf"):
        return StableSelection(index=0, difference=0.0, stable=True)
    return StableSelection(
        index=best_index,
        difference=best_difference,
        stable=False,
    )
