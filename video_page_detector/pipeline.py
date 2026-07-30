from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from .analysis import crop_frame, detect_screen_crop_ratios
from .config import DetectorConfig
from .ffmpeg_io import FFmpegError, FFmpegTools
from .temporal import (
    TemporalFeature,
    find_state_crossover,
    find_temporal_segments,
    make_temporal_feature,
)


class VideoPageDetector:
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
        identifier = video_id or source.stem
        if not identifier or identifier in {".", ".."}:
            raise ValueError("video_id must not be empty")
        if "/" in identifier or "\\" in identifier:
            raise ValueError("video_id cannot contain path separators")

        output_base = Path(output_root)
        run_dir = output_base / identifier
        audit_dir = run_dir / "temporal"
        audit_dir.mkdir(parents=True, exist_ok=True)

        feature_tools = FFmpegTools(
            ffmpeg_path=self.config.ffmpeg_path,
            ffprobe_path=self.config.ffprobe_path,
            analysis_width=self.config.temporal_analysis_width,
            analysis_height=self.config.temporal_analysis_height,
        )
        report("正在读取视频信息", 0.03)
        metadata = feature_tools.probe(source)
        # Analysis frames are intentionally small for speed. Decode final
        # screenshots again at the source video's full resolution.
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
        features: list[TemporalFeature] = []
        chunk_start = 0.0
        chunk_duration = self.config.temporal_chunk_duration_sec
        chunk_count = max(1, math.ceil(metadata.duration_sec / chunk_duration))
        chunk_index = 0
        report(
            f"正在按每 {interval:g} 秒一帧进行时序采样",
            0.08,
        )
        while chunk_start < metadata.duration_sec:
            duration = min(chunk_duration, metadata.duration_sec - chunk_start)
            frames = feature_tools.sample_window(
                source,
                start_sec=chunk_start,
                duration_sec=duration,
                fps=fps,
            )
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
                features.append(
                    make_temporal_feature(
                        screen,
                        timestamp_sec=timestamp,
                        rows=self.config.grid_rows,
                        columns=self.config.grid_columns,
                    )
                )
            chunk_index += 1
            report(
                f"时序采样进度 {chunk_index}/{chunk_count}",
                0.08 + 0.48 * chunk_index / chunk_count,
            )
            chunk_start += duration

        if not features:
            raise FFmpegError("No temporal sample frames could be decoded")

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
        report("正在根据持续相似度自动划分 PPT 页面", 0.62)
        segments = find_temporal_segments(
            features,
            confirmation_samples=confirmation_samples,
            changed_block_ratio=self.config.temporal_changed_block_ratio,
            block_distance_threshold=self.config.block_hash_distance,
            stability_distance=max(12, self.config.block_hash_distance),
            minimum_segment_samples=minimum_segment_samples,
            same_content_similarity=self.config.temporal_same_content_similarity,
        )

        def refine_boundary(segment_index: int) -> float:
            coarse_sec = features[segments[segment_index].start_index].timestamp_sec
            half_window = self.config.temporal_confirmation_sec / 2.0
            window_start = max(0.0, coarse_sec - half_window)
            window_end = min(
                metadata.duration_sec,
                coarse_sec + half_window + interval,
            )
            report(
                f"正在细化换页时间 {segment_index}/{len(segments) - 1}",
                0.62 + 0.12 * segment_index / max(1, len(segments) - 1),
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
            return refined

        refined_starts = [0.0]
        output_pages: list[dict[str, Any]] = []
        segment_audit: list[dict[str, Any]] = []
        for page_index, segment in enumerate(segments):
            if page_index + 1 < len(segments):
                refined_starts.append(refine_boundary(page_index + 1))
            start_sec = refined_starts[page_index]
            end_sec = (
                refined_starts[page_index + 1]
                if page_index + 1 < len(segments)
                else metadata.duration_sec
            )
            representative_sec = features[
                segment.representative_index
            ].timestamp_sec
            report(
                f"正在提取代表截图 {page_index + 1}/{len(segments)}",
                0.62 + 0.31 * (page_index + 1) / max(1, len(segments)),
            )
            representative_frames = output_tools.sample_window(
                source,
                start_sec=representative_sec,
                duration_sec=max(0.6, 1.0 / 2.0),
                fps=2.0,
            )
            if not representative_frames:
                continue
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
                record["note"] = "时序变化处于中间地带，建议人工复核"
            output_pages.append(record)
            if page_ready_callback is not None:
                page_ready_callback(
                    dict(record),
                    page_index + 1,
                    len(segments),
                )
            segment_audit.append(
                {
                    "page_id": page_id,
                    "start_sec": round(start_sec, 3),
                    "end_sec": round(end_sec, 3),
                    "representative_sec": round(representative_sec, 3),
                    "stable_changed_block_ratio": round(
                        segment.change_ratio,
                        4,
                    ),
                    "sample_count": segment.end_index - segment.start_index,
                }
            )

        audit_path = audit_dir / "segments.json"
        audit_payload = {
            "video_id": identifier,
            "mode": "temporal",
            "sample_interval_sec": interval,
            "confirmation_sec": self.config.temporal_confirmation_sec,
            "sample_count": len(features),
            "segment_count": len(output_pages),
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
            "processing_duration_sec": round(time.perf_counter() - started_at, 3),
            "analysis": {
                "mode": "temporal",
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
                "sample_count": len(features),
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
