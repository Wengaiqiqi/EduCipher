"""Legacy FFmpeg scene-threshold detection pipeline."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from .analysis import (
    classify_change,
    crop_frame,
    detect_screen_crop_ratios,
    select_stable_frame,
)
from .config import DetectorConfig
from .ffmpeg_io import FFmpegError, FFmpegTools
from .hashing import compare_grid, hamming_distance, phash
from .models import CandidateAudit, WorkingPage
from .postprocess import (
    merge_intervals,
    merge_short_pages,
    trim_pages_around_no_ppt,
)


def duplicate_page_metrics(
    previous: np.ndarray,
    current: np.ndarray,
    *,
    rows: int,
    columns: int,
    block_hash_distance: int,
    global_hash_distance: int,
    changed_block_ratio: float,
) -> tuple[bool, int, float]:
    """Use global similarity only when the page body is also unchanged."""
    hash_distance = hamming_distance(phash(previous), phash(current))
    changed_ratio, _, _ = compare_grid(
        previous,
        current,
        rows=rows,
        columns=columns,
        changed_distance=block_hash_distance,
    )
    is_duplicate = (
        hash_distance <= global_hash_distance
        and changed_ratio < changed_block_ratio
    )
    return is_duplicate, hash_distance, changed_ratio


class VideoPageDetector:
    def __init__(self, config: DetectorConfig) -> None:
        config.validate()
        self.config = config

    @staticmethod
    def _save_image(frame: np.ndarray, path: Path, quality: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frame).save(path, format="JPEG", quality=quality)

    @staticmethod
    def _display_path(path: Path) -> str:
        return path.as_posix()

    def run(
        self,
        video_path: str | Path,
        *,
        output_root: str | Path = "output",
        video_id: str | None = None,
        progress_callback: Callable[[str, float | None], None] | None = None,
    ) -> dict[str, Any]:
        def report(message: str, progress: float | None) -> None:
            if progress_callback is not None:
                progress_callback(message, progress)

        started_at = time.perf_counter()
        report("正在检查输入和配置", 0.01)
        source = Path(video_path)
        if not source.is_file():
            raise FileNotFoundError(f"Video file does not exist: {source}")
        identifier = video_id or source.stem
        if not identifier or identifier in {".", ".."}:
            raise ValueError("video_id must not be empty")
        if "/" in identifier or "\\" in identifier:
            raise ValueError("video_id cannot contain path separators")

        output_base = Path(output_root)
        run_dir = output_base / identifier
        candidate_dir = run_dir / "candidates"
        candidate_dir.mkdir(parents=True, exist_ok=True)

        tools = FFmpegTools(
            ffmpeg_path=self.config.ffmpeg_path,
            ffprobe_path=self.config.ffprobe_path,
            analysis_width=self.config.analysis_width,
            analysis_height=self.config.analysis_height,
        )
        report("正在读取视频信息", 0.05)
        metadata = tools.probe(source)
        output_tools = FFmpegTools(
            ffmpeg_path=self.config.ffmpeg_path,
            ffprobe_path=self.config.ffprobe_path,
            analysis_width=metadata.width,
            analysis_height=metadata.height,
        )
        crop_ratios = (
            self.config.screen_crop_left_ratio,
            self.config.screen_crop_top_ratio,
            self.config.screen_crop_right_ratio,
            self.config.screen_crop_bottom_ratio,
        )
        if self.config.auto_detect_screen_crop:
            report("正在自动识别投影区域", 0.06)
            calibration_frames = []
            for fraction in (0.1, 0.3, 0.5, 0.7, 0.9):
                sampled = tools.sample_window(
                    source,
                    start_sec=max(0.0, metadata.duration_sec * fraction - 0.5),
                    duration_sec=1.0,
                    fps=1.0,
                )
                if sampled:
                    calibration_frames.append(sampled[0])
            crop_ratios = detect_screen_crop_ratios(
                calibration_frames,
                fallback=crop_ratios,
            )
        thresholds_to_try = [self.config.scene_threshold]
        if (
            self.config.adaptive_scene_detection
            and self.config.scene_threshold_floor < self.config.scene_threshold
        ):
            next_threshold = self.config.scene_threshold / 2.0
            while next_threshold > self.config.scene_threshold_floor:
                thresholds_to_try.append(round(next_threshold, 4))
                next_threshold /= 2.0
            thresholds_to_try.append(self.config.scene_threshold_floor)

        scene_timestamps: list[float] = []
        scene_threshold_used = self.config.scene_threshold
        minimum_candidate_count = max(
            1,
            math.ceil(
                metadata.duration_sec
                / 60.0
                * self.config.minimum_scene_candidates_per_minute
            ),
        )
        for attempt, threshold in enumerate(thresholds_to_try, start=1):
            report(
                (
                    f"正在使用 FFmpeg 粗筛场景变化"
                    f"（阈值 {threshold:g}，第 {attempt} 次）"
                ),
                0.08 + 0.04 * attempt / len(thresholds_to_try),
            )
            scene_timestamps = tools.detect_scene_changes(
                source,
                threshold=threshold,
                crop_ratios=crop_ratios,
            )
            scene_threshold_used = threshold
            if (
                len(scene_timestamps) >= minimum_candidate_count
                or threshold == thresholds_to_try[-1]
            ):
                break
        report(
            (
                f"粗筛完成，共发现 {len(scene_timestamps)} 个候选点，"
                f"实际阈值 {scene_threshold_used:g}"
            ),
            0.25,
        )

        initial_duration = min(
            metadata.duration_sec,
            self.config.stable_delay_sec + 1.0,
        )
        initial_frames = tools.sample_window(
            source,
            start_sec=0.0,
            duration_sec=max(initial_duration, 1.0 / self.config.stable_sample_fps),
            fps=self.config.stable_sample_fps,
        )
        if not initial_frames:
            raise FFmpegError("No video frames could be decoded")
        initial_index = min(
            len(initial_frames) - 1,
            int(round(self.config.stable_delay_sec * self.config.stable_sample_fps)),
        )
        initial_frame = initial_frames[initial_index]
        initial_screen = crop_frame(
            initial_frame,
            left_ratio=crop_ratios[0],
            top_ratio=crop_ratios[1],
            right_ratio=crop_ratios[2],
            bottom_ratio=crop_ratios[3],
        )
        working_pages = [
            WorkingPage(
                start_sec=0.0,
                frame=(
                    initial_screen
                    if self.config.crop_output_screenshots
                    else initial_frame
                ),
                signature=phash(initial_screen),
                confidence="high",
                representative_sec=initial_index / self.config.stable_sample_fps,
                analysis_frame=initial_screen,
            )
        ]

        audits: list[CandidateAudit] = []
        unstable_windows: list[tuple[float, float]] = []
        interval_per_frame = 1.0 / self.config.stable_sample_fps

        for candidate_id, timestamp in enumerate(scene_timestamps, start=1):
            candidate_progress = 0.25 + 0.6 * (
                (candidate_id - 1) / max(1, len(scene_timestamps))
            )
            report(
                (
                    f"正在分析候选点 {candidate_id}/{len(scene_timestamps)}"
                    f"（{timestamp:.1f} 秒）"
                ),
                candidate_progress,
            )
            if timestamp >= metadata.duration_sec:
                continue
            window_start = max(0.0, timestamp - self.config.comparison_offset_sec)
            requested_end = (
                timestamp
                + self.config.stable_delay_sec
                + self.config.stable_window_sec
                + interval_per_frame
            )
            window_end = min(metadata.duration_sec, requested_end)
            frames = tools.sample_window(
                source,
                start_sec=window_start,
                duration_sec=max(window_end - window_start, interval_per_frame),
                fps=self.config.stable_sample_fps,
            )
            if len(frames) < 2:
                audits.append(
                    CandidateAudit(
                        candidate_id=candidate_id,
                        timestamp_sec=timestamp,
                        change_ratio=0.0,
                        mean_hamming_distance=0.0,
                        block_distances=[],
                        classification="unknown",
                        stable=None,
                        stable_frame_sec=None,
                        stable_difference=None,
                        decision="discarded: insufficient frames",
                    )
                )
                continue

            relative_index = (timestamp - window_start) * self.config.stable_sample_fps
            current_index = min(len(frames) - 1, max(1, int(round(relative_index))))
            before_index = max(0, current_index - 1)
            before_frame = frames[before_index]
            current_frame = frames[current_index]
            before_screen = crop_frame(
                before_frame,
                left_ratio=crop_ratios[0],
                top_ratio=crop_ratios[1],
                right_ratio=crop_ratios[2],
                bottom_ratio=crop_ratios[3],
            )
            current_screen = crop_frame(
                current_frame,
                left_ratio=crop_ratios[0],
                top_ratio=crop_ratios[1],
                right_ratio=crop_ratios[2],
                bottom_ratio=crop_ratios[3],
            )

            file_prefix = f"candidate_{candidate_id:04d}"
            before_path = candidate_dir / f"{file_prefix}_before.jpg"
            current_path = candidate_dir / f"{file_prefix}_at.jpg"
            self._save_image(before_frame, before_path, self.config.jpeg_quality)
            self._save_image(current_frame, current_path, self.config.jpeg_quality)
            files = {
                "before": self._display_path(before_path),
                "at": self._display_path(current_path),
            }

            ratio, mean_distance, distances = compare_grid(
                before_screen,
                current_screen,
                rows=self.config.grid_rows,
                columns=self.config.grid_columns,
                changed_distance=self.config.block_hash_distance,
            )
            confidence = classify_change(
                ratio,
                low_threshold=self.config.low_change_ratio,
                high_threshold=self.config.high_change_ratio,
            )
            if confidence is None:
                audits.append(
                    CandidateAudit(
                        candidate_id=candidate_id,
                        timestamp_sec=timestamp,
                        change_ratio=ratio,
                        mean_hamming_distance=mean_distance,
                        block_distances=distances,
                        classification="not_change",
                        stable=None,
                        stable_frame_sec=None,
                        stable_difference=None,
                        decision="discarded: local change or camera motion",
                        files=files,
                    )
                )
                continue

            stable_start_index = int(
                math.ceil(
                    (
                        timestamp
                        + self.config.stable_delay_sec
                        - window_start
                    )
                    * self.config.stable_sample_fps
                )
            )
            analysis_frames = [
                crop_frame(
                    frame,
                    left_ratio=crop_ratios[0],
                    top_ratio=crop_ratios[1],
                    right_ratio=crop_ratios[2],
                    bottom_ratio=crop_ratios[3],
                )
                for frame in frames
            ]
            selection = select_stable_frame(
                analysis_frames,
                first_index=stable_start_index,
                threshold=self.config.stable_diff_threshold,
            )
            representative = frames[selection.index]
            representative_screen = analysis_frames[selection.index]
            representative_sec = min(
                metadata.duration_sec,
                window_start + selection.index * interval_per_frame,
            )
            stable_path = candidate_dir / f"{file_prefix}_stable.jpg"
            self._save_image(representative, stable_path, self.config.jpeg_quality)
            files["stable"] = self._display_path(stable_path)

            note_parts: list[str] = []
            if confidence == "low":
                note_parts.append(
                    "变化区域占比处于中间地带，建议人工复核"
                )
            if not selection.stable:
                confidence = "low"
                note_parts.append(
                    "稳定窗口内未找到稳定帧，已选取帧差最小画面"
                )
                unstable_windows.append((timestamp, window_end))

            signature = phash(representative_screen)
            previous_analysis = working_pages[-1].analysis_frame
            duplicate_distance = hamming_distance(
                working_pages[-1].signature,
                signature,
            )
            duplicate_change_ratio = 1.0
            is_duplicate = False
            if previous_analysis is not None:
                (
                    is_duplicate,
                    duplicate_distance,
                    duplicate_change_ratio,
                ) = duplicate_page_metrics(
                    previous_analysis,
                    representative_screen,
                    rows=self.config.grid_rows,
                    columns=self.config.grid_columns,
                    block_hash_distance=self.config.block_hash_distance,
                    global_hash_distance=self.config.duplicate_hash_distance,
                    changed_block_ratio=(
                        self.config.duplicate_changed_block_ratio
                    ),
                )
            if is_duplicate:
                audits.append(
                    CandidateAudit(
                        candidate_id=candidate_id,
                        timestamp_sec=timestamp,
                        change_ratio=ratio,
                        mean_hamming_distance=mean_distance,
                        block_distances=distances,
                        classification=confidence,
                        stable=selection.stable,
                        stable_frame_sec=representative_sec,
                        stable_difference=selection.difference,
                        decision=(
                            "discarded: duplicate of previous confirmed page "
                            f"(global_distance={duplicate_distance}, "
                            f"changed_blocks={duplicate_change_ratio:.3f})"
                        ),
                        duplicate_hash_distance=duplicate_distance,
                        duplicate_changed_block_ratio=duplicate_change_ratio,
                        files=files,
                    )
                )
                continue

            working_pages.append(
                WorkingPage(
                    start_sec=timestamp,
                    frame=(
                        representative_screen
                        if self.config.crop_output_screenshots
                        else representative
                    ),
                    signature=signature,
                    confidence=confidence,
                    note="；".join(note_parts) if note_parts else None,
                    representative_sec=representative_sec,
                    analysis_frame=representative_screen,
                )
            )
            audits.append(
                CandidateAudit(
                    candidate_id=candidate_id,
                    timestamp_sec=timestamp,
                    change_ratio=ratio,
                    mean_hamming_distance=mean_distance,
                    block_distances=distances,
                    classification=confidence,
                    stable=selection.stable,
                    stable_frame_sec=representative_sec,
                    stable_difference=selection.difference,
                    decision=(
                        "accepted as provisional page "
                        f"(global_distance={duplicate_distance}, "
                        f"changed_blocks={duplicate_change_ratio:.3f})"
                    ),
                    duplicate_hash_distance=duplicate_distance,
                    duplicate_changed_block_ratio=duplicate_change_ratio,
                    files=files,
                )
            )

        report("正在执行去重、短页面合并和无 PPT 区间整理", 0.88)
        no_ppt_intervals = merge_intervals(
            unstable_windows,
            max_gap=self.config.no_ppt_merge_gap_sec,
            min_duration=self.config.no_ppt_min_duration_sec,
        )
        no_ppt_intervals = [
            (start, min(end, metadata.duration_sec))
            for start, end in no_ppt_intervals
        ]
        pages_after_duration_filter = merge_short_pages(
            working_pages,
            video_duration=metadata.duration_sec,
            min_duration=self.config.min_page_duration_sec,
        )
        page_intervals = [
            (
                page.start_sec,
                (
                    pages_after_duration_filter[index + 1].start_sec
                    if index + 1 < len(pages_after_duration_filter)
                    else metadata.duration_sec
                ),
                page,
            )
            for index, page in enumerate(pages_after_duration_filter)
        ]
        page_intervals = trim_pages_around_no_ppt(
            page_intervals,
            no_ppt_intervals,
        )

        report("正在保存代表截图和结果文件", 0.94)
        output_pages: list[dict[str, Any]] = []
        for page_id, (start, end, page) in enumerate(page_intervals, start=1):
            representative_frames = output_tools.sample_window(
                source,
                start_sec=page.representative_sec,
                duration_sec=0.6,
                fps=2.0,
            )
            if representative_frames:
                full_frame = representative_frames[0]
                screen_frame = crop_frame(
                    full_frame,
                    left_ratio=crop_ratios[0],
                    top_ratio=crop_ratios[1],
                    right_ratio=crop_ratios[2],
                    bottom_ratio=crop_ratios[3],
                )
                output_frame = (
                    screen_frame
                    if self.config.crop_output_screenshots
                    else full_frame
                )
            else:
                output_frame = page.frame
            screenshot_path = run_dir / f"page_{page_id:03d}.jpg"
            self._save_image(
                output_frame,
                screenshot_path,
                self.config.jpeg_quality,
            )
            record: dict[str, Any] = {
                "page_id": page_id,
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
                "representative_sec": round(page.representative_sec, 3),
                "confidence": page.confidence,
                "screenshot_path": self._display_path(screenshot_path),
            }
            if page.note:
                record["note"] = page.note
            output_pages.append(record)

        audit_path = candidate_dir / "candidates.json"
        audit_payload = {
            "video_id": identifier,
            "scene_candidate_count": len(scene_timestamps),
            "scene_threshold_requested": self.config.scene_threshold,
            "scene_threshold_used": scene_threshold_used,
            "minimum_scene_candidate_target": minimum_candidate_count,
            "screen_crop_ratios": {
                "left": crop_ratios[0],
                "top": crop_ratios[1],
                "right": crop_ratios[2],
                "bottom": crop_ratios[3],
            },
            "adaptive_fallback_applied": (
                scene_threshold_used != self.config.scene_threshold
            ),
            "candidates": [audit.to_dict() for audit in audits],
        }
        audit_path.write_text(
            json.dumps(audit_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        result = {
            "video_id": identifier,
            "video_duration_sec": round(metadata.duration_sec, 3),
            "processing_duration_sec": round(time.perf_counter() - started_at, 3),
            "analysis": {
                "source_resolution": {
                    "width": metadata.width,
                    "height": metadata.height,
                },
                "screenshot_resolution": (
                    {
                        "width": round(
                            metadata.width
                            * (1.0 - crop_ratios[0] - crop_ratios[2])
                        ),
                        "height": round(
                            metadata.height
                            * (1.0 - crop_ratios[1] - crop_ratios[3])
                        ),
                    }
                    if self.config.crop_output_screenshots
                    else {
                        "width": metadata.width,
                        "height": metadata.height,
                    }
                ),
                "scene_candidate_count": len(scene_timestamps),
                "scene_threshold_requested": self.config.scene_threshold,
                "scene_threshold_used": scene_threshold_used,
                "minimum_scene_candidate_target": minimum_candidate_count,
                "screen_crop_ratios": {
                    "left": crop_ratios[0],
                    "top": crop_ratios[1],
                    "right": crop_ratios[2],
                    "bottom": crop_ratios[3],
                },
                "adaptive_fallback_applied": (
                    scene_threshold_used != self.config.scene_threshold
                ),
            },
            "pages": output_pages,
            "no_ppt_segments": [
                {
                    "start_sec": round(start, 3),
                    "end_sec": round(end, 3),
                    "reason": "画面持续大面积变化或无法稳定，建议人工复核",
                }
                for start, end in no_ppt_intervals
            ],
            "artifacts": {
                "candidate_audit_path": self._display_path(audit_path),
            },
            "config": self.config.to_dict(),
        }
        result_path = run_dir / "result.json"
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        report("处理完成", 1.0)
        return result
