from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from .analysis import crop_frame, detect_screen_crop_ratios
from .config import DetectorConfig
from .ffmpeg_io import FFmpegError, FFmpegTools
from .output_paths import resolve_run_directory, validate_video_id
from .temporal import (
    IncrementalTemporalSegmenter,
    TemporalFeature,
    TemporalSegment,
    find_state_crossover,
    make_temporal_feature,
)


class VideoPageDetector:
    """Production temporal detector with incremental page confirmation."""

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config

    def run(
        self,
        video_path: str | Path,
        *,
        output_root: str | Path = "output",
        video_id: str | None = None,
        progress_callback: Callable[[str, float | None], None] | None = None,
        page_ready_callback: (
            Callable[[dict[str, Any], int, int], None] | None
        ) = None,
    ) -> dict[str, Any]:
        def report(message: str, progress: float | None) -> None:
            if progress_callback is not None:
                progress_callback(message, progress)

        started_at = time.perf_counter()
        source = Path(video_path)
        if not source.is_file():
            raise FileNotFoundError(f"Video file does not exist: {source}")
        identifier = validate_video_id(video_id or source.stem)
        run_dir = resolve_run_directory(output_root, identifier)
        audit_dir = run_dir / "temporal"
        audit_dir.mkdir(parents=True, exist_ok=True)
        feature_tools = FFmpegTools(
            ffmpeg_path=self.config.ffmpeg_path,
            ffprobe_path=self.config.ffprobe_path,
            analysis_width=self.config.temporal_analysis_width,
            analysis_height=self.config.temporal_analysis_height,
        )
        report("正在读取视频信息", 0.03)
        sys.stderr.write(f"[DEBUG] probing video: {source}\n")
        sys.stderr.flush()
        metadata = feature_tools.probe(source)
        output_tools = FFmpegTools(
            ffmpeg_path=self.config.ffmpeg_path,
            ffprobe_path=self.config.ffprobe_path,
            analysis_width=metadata.width,
            analysis_height=metadata.height,
        )

        interval = self.config.temporal_sample_interval_sec
        fps = 1.0 / interval
        fallback_crop = (
            self.config.screen_crop_left_ratio,
            self.config.screen_crop_top_ratio,
            self.config.screen_crop_right_ratio,
            self.config.screen_crop_bottom_ratio,
        )
        crop_ratios = fallback_crop
        if self.config.auto_detect_screen_crop:
            report("正在自动识别投影区域", 0.05)
            calibration_frames = []
            for fraction in (0.1, 0.3, 0.5, 0.7, 0.9):
                calibration = feature_tools.sample_window(
                    source,
                    start_sec=max(
                        0.0,
                        metadata.duration_sec * fraction - 0.5,
                    ),
                    duration_sec=1.0,
                    fps=1.0,
                )
                if calibration:
                    calibration_frames.append(calibration[0])
            crop_ratios = detect_screen_crop_ratios(
                calibration_frames,
                fallback=fallback_crop,
            )

        confirmation_samples = max(
            2,
            math.ceil(
                self.config.temporal_confirmation_sec
                / self.config.temporal_sample_interval_sec
            ),
        )
        minimum_segment_samples = max(
            2,
            math.ceil(
                self.config.min_page_duration_sec
                / self.config.temporal_sample_interval_sec
            ),
        )
        segmenter = IncrementalTemporalSegmenter(
            confirmation_samples=confirmation_samples,
            changed_block_ratio=self.config.temporal_changed_block_ratio,
            block_distance_threshold=self.config.block_hash_distance,
            stability_distance=max(12, self.config.block_hash_distance),
            minimum_segment_samples=minimum_segment_samples,
            same_content_similarity=(
                self.config.temporal_same_content_similarity
            ),
            retained_tail_segments=2,
        )
        output_pages: list[dict[str, Any]] = []
        segment_audit: list[dict[str, Any]] = []
        refined_boundaries: dict[int, float] = {}
        first_page_ready_after_sec: float | None = None

        def refine_boundary(
            segment_index: int,
            segments: list[TemporalSegment],
        ) -> float:
            cached = refined_boundaries.get(segment_index)
            if cached is not None:
                return cached
            features = segmenter.features
            coarse_sec = features[
                segments[segment_index].start_index
            ].timestamp_sec
            half_window = self.config.temporal_confirmation_sec / 2.0
            window_start = max(0.0, coarse_sec - half_window)
            window_end = min(
                metadata.duration_sec,
                coarse_sec + half_window + interval,
            )
            report(
                f"正在提前细化换页时间 {segment_index}",
                None,
            )
            boundary_frames = feature_tools.sample_window(
                source,
                start_sec=window_start,
                duration_sec=max(
                    window_end - window_start,
                    1.0 / self.config.temporal_refinement_fps,
                ),
                fps=self.config.temporal_refinement_fps,
            )
            boundary_features: list[TemporalFeature] = []
            for frame_index, frame in enumerate(boundary_frames):
                screen = crop_frame(
                    frame,
                    left_ratio=crop_ratios[0],
                    top_ratio=crop_ratios[1],
                    right_ratio=crop_ratios[2],
                    bottom_ratio=crop_ratios[3],
                )
                boundary_features.append(
                    make_temporal_feature(
                        screen,
                        timestamp_sec=(
                            window_start
                            + frame_index / self.config.temporal_refinement_fps
                        ),
                        rows=self.config.grid_rows,
                        columns=self.config.grid_columns,
                    )
                )
            crossover = find_state_crossover(
                boundary_features,
                before_state=features[
                    segments[segment_index - 1].representative_index
                ],
                after_state=features[
                    segments[segment_index].representative_index
                ],
                persistence=max(
                    2,
                    round(self.config.temporal_refinement_fps * 0.5),
                ),
            )
            refined = (
                boundary_features[crossover].timestamp_sec
                if crossover is not None
                else coarse_sec
            )
            refined_boundaries[segment_index] = refined
            return refined

        def emit_confirmed_pages(
            newly_confirmed: list[TemporalSegment],
            segments: list[TemporalSegment],
            *,
            final: bool,
        ) -> None:
            nonlocal first_page_ready_after_sec
            first_index = len(output_pages)
            expected = segments[
                first_index : first_index + len(newly_confirmed)
            ]
            if expected != newly_confirmed:
                raise RuntimeError("增量页面顺序与稳定前缀不一致。")
            for offset, segment in enumerate(newly_confirmed):
                segment_index = first_index + offset
                start_sec = (
                    0.0
                    if segment_index == 0
                    else refine_boundary(segment_index, segments)
                )
                is_last = (
                    final and segment_index == len(segments) - 1
                )
                end_sec = (
                    metadata.duration_sec
                    if is_last
                    else refine_boundary(segment_index + 1, segments)
                )
                representative_sec = segmenter.features[
                    segment.representative_index
                ].timestamp_sec
                report(
                    f"正在提前提取代表截图 {segment_index + 1}",
                    None,
                )
                representative_frames = output_tools.sample_window(
                    source,
                    start_sec=representative_sec,
                    duration_sec=0.6,
                    fps=2.0,
                )
                if not representative_frames:
                    raise FFmpegError("无法提取已确认页面的高清截图")
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
                page_id = len(output_pages) + 1
                screenshot_path = run_dir / f"page_{page_id:03d}.jpg"
                Image.fromarray(output_frame).save(
                    screenshot_path,
                    format="JPEG",
                    quality=self.config.jpeg_quality,
                )
                record: dict[str, Any] = {
                    "page_id": page_id,
                    "start_sec": round(start_sec, 3),
                    "end_sec": round(end_sec, 3),
                    "representative_sec": round(representative_sec, 3),
                    "confidence": segment.confidence,
                    "screenshot_path": screenshot_path.as_posix(),
                }
                if segment.confidence != "high":
                    record["note"] = (
                        "时序变化处于中间地带，建议人工复核"
                    )
                output_pages.append(record)
                if first_page_ready_after_sec is None:
                    first_page_ready_after_sec = (
                        time.perf_counter() - started_at
                    )
                if page_ready_callback is not None:
                    callback_total = (
                        len(segments)
                        if final
                        else len(segmenter.confirmed_segments)
                    )
                    page_ready_callback(
                        dict(record),
                        page_id,
                        max(page_id, callback_total),
                    )
                segment_audit.append(
                    {
                        "page_id": page_id,
                        "start_sec": round(start_sec, 3),
                        "end_sec": round(end_sec, 3),
                        "representative_sec": round(
                            representative_sec,
                            3,
                        ),
                        "stable_changed_block_ratio": round(
                            segment.change_ratio,
                            4,
                        ),
                        "sample_count": (
                            segment.end_index - segment.start_index
                        ),
                    }
                )

        chunk_start = 0.0
        chunk_duration = self.config.temporal_chunk_duration_sec
        chunk_count = max(
            1,
            math.ceil(metadata.duration_sec / chunk_duration),
        )
        chunk_index = 0
        report(
            f"正在按每 {interval:g} 秒一帧流式分析",
            0.08,
        )
        while chunk_start < metadata.duration_sec:
            duration = min(
                chunk_duration,
                metadata.duration_sec - chunk_start,
            )
            frames = feature_tools.sample_window(
                source,
                start_sec=chunk_start,
                duration_sec=duration,
                fps=fps,
            )
            chunk_features: list[TemporalFeature] = []
            for frame_index, frame in enumerate(frames):
                timestamp = chunk_start + frame_index * interval
                if timestamp >= metadata.duration_sec:
                    break
                screen = crop_frame(
                    frame,
                    left_ratio=crop_ratios[0],
                    top_ratio=crop_ratios[1],
                    right_ratio=crop_ratios[2],
                    bottom_ratio=crop_ratios[3],
                )
                chunk_features.append(
                    make_temporal_feature(
                        screen,
                        timestamp_sec=timestamp,
                        rows=self.config.grid_rows,
                        columns=self.config.grid_columns,
                    )
                )
            chunk_index += 1
            is_final = (
                chunk_start + duration
                >= metadata.duration_sec - 1e-6
            )
            newly_confirmed, segments = segmenter.extend(
                chunk_features,
                final=is_final,
            )
            emit_confirmed_pages(
                newly_confirmed,
                segments,
                final=is_final,
            )
            report(
                (
                    f"流式时序进度 {chunk_index}/{chunk_count}，"
                    f"已确认 {len(output_pages)} 页"
                ),
                0.08 + 0.85 * chunk_index / chunk_count,
            )
            chunk_start += duration

        if not segmenter.features:
            raise FFmpegError("No temporal sample frames could be decoded")
        final_segments = segmenter.latest_segments
        if len(output_pages) != len(final_segments):
            raise RuntimeError("流式时序页面没有全部输出。")

        audit_path = audit_dir / "segments.json"
        audit_payload = {
            "video_id": identifier,
            "mode": "temporal",
            "streaming_page_confirmation": True,
            "retained_tail_segments": 2,
            "sample_interval_sec": interval,
            "confirmation_sec": self.config.temporal_confirmation_sec,
            "sample_count": len(segmenter.features),
            "segment_count": len(output_pages),
            "first_page_ready_after_sec": (
                round(first_page_ready_after_sec, 3)
                if first_page_ready_after_sec is not None
                else None
            ),
            "screen_crop_ratios": {
                "left": crop_ratios[0],
                "top": crop_ratios[1],
                "right": crop_ratios[2],
                "bottom": crop_ratios[3],
            },
            "segments": segment_audit,
        }
        audit_path.write_text(
            json.dumps(audit_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        result = {
            "video_id": identifier,
            "video_duration_sec": round(metadata.duration_sec, 3),
            "processing_duration_sec": round(
                time.perf_counter() - started_at,
                3,
            ),
            "analysis": {
                "mode": "temporal",
                "streaming_page_confirmation": True,
                "retained_tail_segments": 2,
                "first_page_ready_after_sec": (
                    round(first_page_ready_after_sec, 3)
                    if first_page_ready_after_sec is not None
                    else None
                ),
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
                "sample_interval_sec": interval,
                "confirmation_sec": self.config.temporal_confirmation_sec,
                "sample_count": len(segmenter.features),
                "segment_count": len(output_pages),
                "screen_crop_ratios": {
                    "left": crop_ratios[0],
                    "top": crop_ratios[1],
                    "right": crop_ratios[2],
                    "bottom": crop_ratios[3],
                },
                "representative_strategy": (
                    "maximum information gain within the same temporal segment"
                ),
            },
            "pages": output_pages,
            "artifacts": {
                "temporal_audit_path": audit_path.as_posix(),
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


# Backward-compatible name used by the experiment reproduction scripts.
StreamingVideoPageDetector = VideoPageDetector
