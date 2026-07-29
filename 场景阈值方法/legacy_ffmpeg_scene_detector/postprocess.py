"""Post-processing rules used only by the legacy scene detector."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from .models import WorkingPage


def merge_intervals(
    intervals: Iterable[tuple[float, float]],
    *,
    max_gap: float,
    min_duration: float,
) -> list[tuple[float, float]]:
    normalized = sorted(
        (max(0.0, float(start)), max(0.0, float(end)))
        for start, end in intervals
        if end > start
    )
    if not normalized:
        return []
    merged: list[list[float]] = [[normalized[0][0], normalized[0][1]]]
    for start, end in normalized[1:]:
        current = merged[-1]
        if start <= current[1] + max_gap:
            current[1] = max(current[1], end)
        else:
            merged.append([start, end])
    return [
        (start, end)
        for start, end in merged
        if end - start >= min_duration
    ]


def merge_short_pages(
    pages: list[WorkingPage],
    *,
    video_duration: float,
    min_duration: float,
) -> list[WorkingPage]:
    result = [replace(page) for page in sorted(pages, key=lambda page: page.start_sec)]
    while len(result) > 1:
        short_index: int | None = None
        for index, page in enumerate(result):
            end = (
                result[index + 1].start_sec
                if index + 1 < len(result)
                else video_duration
            )
            if end - page.start_sec < min_duration:
                short_index = index
                break
        if short_index is None:
            break
        if short_index == 0:
            result[1].start_sec = 0.0
            result.pop(0)
        else:
            result.pop(short_index)
    return result


def trim_pages_around_no_ppt(
    intervals: list[tuple[float, float, WorkingPage]],
    no_ppt_segments: list[tuple[float, float]],
) -> list[tuple[float, float, WorkingPage]]:
    result: list[tuple[float, float, WorkingPage]] = []
    for original_start, original_end, page in intervals:
        start, end = original_start, original_end
        for gap_start, gap_end in no_ppt_segments:
            if gap_end <= start or gap_start >= end:
                continue
            if gap_start <= start < gap_end:
                start = gap_end
            elif start < gap_start < end:
                end = gap_start
            if end <= start:
                break
        if end > start:
            result.append((start, end, page))
    return result
