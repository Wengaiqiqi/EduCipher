from __future__ import annotations

import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .ffmpeg_io import FFmpegError, FFmpegTools


_PTS_TIME = re.compile(r"pts_time:\s*([0-9]+(?:\.[0-9]+)?)")


@dataclass(frozen=True)
class SceneHint:
    timestamp_sec: float
    emitted_after_sec: float


@dataclass(frozen=True)
class BoundaryMatch:
    temporal_sec: float
    scene_sec: float | None
    error_sec: float | None
    matched: bool


@dataclass(frozen=True)
class PageReuseDecision:
    page_id: int
    start_sec: float
    end_sec: float
    reusable: bool
    provisional_start_sec: float | None
    provisional_end_sec: float | None
    start_error_sec: float | None
    end_error_sec: float | None
    reason: str


SceneHintCallback = Callable[[SceneHint], None]


def parse_scene_timestamp(line: str) -> float | None:
    match = _PTS_TIME.search(line)
    if match is None:
        return None
    timestamp = float(match.group(1))
    return timestamp if timestamp > 0 else None


def filter_short_scene_intervals(
    hints: Iterable[SceneHint],
    *,
    min_interval_sec: float,
) -> list[SceneHint]:
    if min_interval_sec < 0:
        raise ValueError("min_interval_sec不能为负数。")
    accepted: list[SceneHint] = []
    previous_boundary = 0.0
    for hint in sorted(hints, key=lambda item: item.timestamp_sec):
        if hint.timestamp_sec - previous_boundary < min_interval_sec:
            continue
        accepted.append(hint)
        previous_boundary = hint.timestamp_sec
    return accepted


def scene_filter_graph(
    threshold: float,
    crop_ratios: tuple[float, float, float, float] | None,
) -> str:
    filters: list[str] = []
    if crop_ratios is not None:
        left, top, right, bottom = crop_ratios
        width_ratio = 1.0 - left - right
        height_ratio = 1.0 - top - bottom
        filters.append(
            (
                f"crop=iw*{width_ratio:.6f}:ih*{height_ratio:.6f}:"
                f"iw*{left:.6f}:ih*{top:.6f}"
            )
        )
    filters.extend([f"select=gt(scene\\,{threshold})", "showinfo"])
    return ",".join(filters)


def stream_scene_hints(
    video_path: str | Path,
    *,
    threshold: float = 0.05,
    crop_ratios: tuple[float, float, float, float] | None = None,
    ffmpeg_path: str | None = None,
    callback: SceneHintCallback | None = None,
) -> list[SceneHint]:
    source = Path(video_path)
    if not source.is_file():
        raise FileNotFoundError(f"视频文件不存在：{source}")
    tools = FFmpegTools(
        ffmpeg_path=ffmpeg_path,
        ffprobe_path=None,
        analysis_width=320,
        analysis_height=180,
    )
    command = [
        tools.ffmpeg,
        "-hide_banner",
        "-nostats",
        "-loglevel",
        "info",
        "-i",
        str(source),
        "-vf",
        scene_filter_graph(threshold, crop_ratios),
        "-an",
        "-sn",
        "-f",
        "null",
        "-",
    ]
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stderr is not None
    hints: list[SceneHint] = []
    seen: set[float] = set()
    tail: list[str] = []
    for line in process.stderr:
        tail.append(line.rstrip())
        if len(tail) > 12:
            tail.pop(0)
        timestamp = parse_scene_timestamp(line)
        if timestamp is None or timestamp in seen:
            continue
        seen.add(timestamp)
        hint = SceneHint(
            timestamp_sec=timestamp,
            emitted_after_sec=time.perf_counter() - started,
        )
        hints.append(hint)
        if callback is not None:
            callback(hint)
    return_code = process.wait()
    if return_code != 0:
        raise FFmpegError(
            "FFmpeg场景提示检测失败：\n" + "\n".join(tail)
        )
    return sorted(hints, key=lambda item: item.timestamp_sec)


def _match_boundaries(
    temporal_boundaries: Sequence[float],
    scene_boundaries: Sequence[float],
    *,
    tolerance_sec: float,
) -> list[BoundaryMatch]:
    available = set(range(len(scene_boundaries)))
    matches: list[BoundaryMatch] = []
    for temporal in temporal_boundaries:
        candidates = [
            (abs(scene_boundaries[index] - temporal), index)
            for index in available
            if abs(scene_boundaries[index] - temporal) <= tolerance_sec
        ]
        if not candidates:
            matches.append(
                BoundaryMatch(
                    temporal_sec=temporal,
                    scene_sec=None,
                    error_sec=None,
                    matched=False,
                )
            )
            continue
        error, index = min(candidates)
        available.remove(index)
        matches.append(
            BoundaryMatch(
                temporal_sec=temporal,
                scene_sec=scene_boundaries[index],
                error_sec=error,
                matched=True,
            )
        )
    return matches


def analyze_reuse(
    temporal_pages: Sequence[Mapping[str, Any]],
    scene_hints: Sequence[SceneHint],
    *,
    video_duration_sec: float,
    tolerance_sec: float = 2.0,
) -> dict[str, Any]:
    pages = [dict(page) for page in temporal_pages]
    if not pages:
        raise ValueError("时序算法没有输出页面。")
    scene_boundaries = sorted(
        {
            round(hint.timestamp_sec, 6)
            for hint in scene_hints
            if 0 < hint.timestamp_sec < video_duration_sec
        }
    )
    temporal_boundaries = [
        float(page["start_sec"])
        for page in pages[1:]
    ]
    boundary_matches = _match_boundaries(
        temporal_boundaries,
        scene_boundaries,
        tolerance_sec=tolerance_sec,
    )
    matched_scene_by_temporal = {
        round(match.temporal_sec, 6): match.scene_sec
        for match in boundary_matches
        if match.matched
    }
    provisional_boundaries = [0.0, *scene_boundaries, video_duration_sec]
    provisional_intervals = [
        (provisional_boundaries[index], provisional_boundaries[index + 1])
        for index in range(len(provisional_boundaries) - 1)
    ]

    decisions: list[PageReuseDecision] = []
    for page in pages:
        page_id = int(page["page_id"])
        start = float(page["start_sec"])
        end = float(page["end_sec"])
        provisional_start = (
            0.0
            if start == 0.0
            else matched_scene_by_temporal.get(round(start, 6))
        )
        provisional_end = (
            video_duration_sec
            if abs(end - video_duration_sec) <= 0.001
            else matched_scene_by_temporal.get(round(end, 6))
        )
        start_error = (
            abs(provisional_start - start)
            if provisional_start is not None
            else None
        )
        end_error = (
            abs(provisional_end - end)
            if provisional_end is not None
            else None
        )
        interval_exists = (
            provisional_start is not None
            and provisional_end is not None
            and any(
                abs(candidate_start - provisional_start) <= 0.001
                and abs(candidate_end - provisional_end) <= 0.001
                for candidate_start, candidate_end in provisional_intervals
            )
        )
        reusable = bool(
            interval_exists
            and start_error is not None
            and end_error is not None
            and start_error <= tolerance_sec
            and end_error <= tolerance_sec
        )
        if reusable:
            reason = "场景区间与时序区间两端均匹配"
        elif provisional_start is None or provisional_end is None:
            reason = "至少一个时序边界没有场景候选"
        else:
            reason = "页面内部存在额外场景候选，或边界误差超限"
        decisions.append(
            PageReuseDecision(
                page_id=page_id,
                start_sec=start,
                end_sec=end,
                reusable=reusable,
                provisional_start_sec=provisional_start,
                provisional_end_sec=provisional_end,
                start_error_sec=start_error,
                end_error_sec=end_error,
                reason=reason,
            )
        )

    matched_boundaries = sum(match.matched for match in boundary_matches)
    reusable_pages = sum(item.reusable for item in decisions)
    first_hint = min(
        scene_hints,
        key=lambda item: item.emitted_after_sec,
        default=None,
    )
    return {
        "scene_hint_count": len(scene_boundaries),
        "temporal_boundary_count": len(temporal_boundaries),
        "matched_boundary_count": matched_boundaries,
        "boundary_match_rate": round(
            matched_boundaries / max(len(temporal_boundaries), 1),
            4,
        ),
        "reusable_page_count": reusable_pages,
        "reprocess_page_count": len(decisions) - reusable_pages,
        "page_reuse_rate": round(
            reusable_pages / max(len(decisions), 1),
            4,
        ),
        "first_scene_hint_emitted_after_sec": (
            round(first_hint.emitted_after_sec, 3)
            if first_hint is not None
            else None
        ),
        "last_scene_hint_emitted_after_sec": (
            round(
                max(item.emitted_after_sec for item in scene_hints),
                3,
            )
            if scene_hints
            else None
        ),
        "scene_hints": [asdict(item) for item in scene_hints],
        "boundary_matches": [asdict(item) for item in boundary_matches],
        "page_decisions": [asdict(item) for item in decisions],
    }


def load_temporal_result(path: str | Path) -> dict[str, Any]:
    import json

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload.get("pages"), list):
        raise ValueError("result.json缺少pages数组。")
    return payload
