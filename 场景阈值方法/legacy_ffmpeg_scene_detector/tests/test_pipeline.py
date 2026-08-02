import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from legacy_ffmpeg_scene_detector.config import DetectorConfig
from legacy_ffmpeg_scene_detector.ffmpeg_io import VideoMetadata
from legacy_ffmpeg_scene_detector.pipeline import (
    VideoPageDetector,
    duplicate_page_metrics,
)


class DuplicatePageTests(unittest.TestCase):
    def test_identical_frames_are_duplicates(self) -> None:
        frame = np.random.default_rng(9).integers(
            0,
            256,
            size=(120, 160, 3),
            dtype=np.uint8,
        )
        duplicate, _, changed_ratio = duplicate_page_metrics(
            frame,
            frame.copy(),
            rows=3,
            columns=4,
            block_hash_distance=10,
            global_hash_distance=6,
            changed_block_ratio=0.25,
        )
        self.assertTrue(duplicate)
        self.assertEqual(changed_ratio, 0.0)

    def test_same_template_with_changed_body_is_not_duplicate(self) -> None:
        before = np.full((120, 160, 3), 245, dtype=np.uint8)
        after = before.copy()
        before[:16] = (210, 40, 30)
        after[:16] = (210, 40, 30)
        rng = np.random.default_rng(10)
        before[30:105, 15:75] = rng.integers(
            0,
            256,
            size=(75, 60, 3),
            dtype=np.uint8,
        )
        after[30:105, 85:145] = rng.integers(
            0,
            256,
            size=(75, 60, 3),
            dtype=np.uint8,
        )
        duplicate, _, changed_ratio = duplicate_page_metrics(
            before,
            after,
            rows=3,
            columns=4,
            block_hash_distance=10,
            global_hash_distance=64,
            changed_block_ratio=0.25,
        )
        self.assertFalse(duplicate)
        self.assertGreaterEqual(changed_ratio, 0.25)


class _FakeFFmpegTools:
    def __init__(
        self,
        *,
        analysis_width: int,
        analysis_height: int,
        **_options: object,
    ) -> None:
        rng = np.random.default_rng(44)
        shape = (analysis_height, analysis_width, 3)
        self.first = rng.integers(0, 256, size=shape, dtype=np.uint8)
        self.second = rng.integers(0, 256, size=shape, dtype=np.uint8)

    def probe(self, _: Path) -> VideoMetadata:
        return VideoMetadata(duration_sec=20.0, width=1280, height=720)

    def detect_scene_changes(
        self,
        _: Path,
        *,
        threshold: float,
        crop_ratios: tuple[float, float, float, float] | None = None,
        **_options: object,
    ) -> list[float]:
        self.last_threshold = threshold
        self.last_crop_ratios = crop_ratios
        return [10.0]

    def sample_window(
        self,
        _: Path,
        *,
        start_sec: float,
        duration_sec: float,
        fps: float,
    ) -> list[np.ndarray]:
        frame_count = max(2, int(duration_sec * fps))
        if start_sec < 1:
            return [self.first.copy() for _ in range(frame_count)]
        return [
            *(self.first.copy() for _ in range(2)),
            *(self.second.copy() for _ in range(max(0, frame_count - 2))),
        ]


class _FallbackFFmpegTools(_FakeFFmpegTools):
    thresholds: list[float] = []

    def detect_scene_changes(
        self,
        _: Path,
        *,
        threshold: float,
        crop_ratios: tuple[float, float, float, float] | None = None,
        **_options: object,
    ) -> list[float]:
        self.__class__.thresholds.append(threshold)
        return [10.0] if threshold <= 0.05 else []


class PipelineIntegrationTests(unittest.TestCase):
    @patch("legacy_ffmpeg_scene_detector.pipeline.FFmpegTools", _FakeFFmpegTools)
    def test_writes_result_screenshots_and_candidate_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "class.mp4"
            video.write_bytes(b"fake")
            output = root / "output"
            config = DetectorConfig(
                auto_detect_screen_crop=False,
                analysis_width=64,
                analysis_height=36,
            )
            progress_events: list[tuple[str, float | None]] = []
            result = VideoPageDetector(config).run(
                video,
                output_root=output,
                video_id="class_01",
                progress_callback=lambda message, progress: progress_events.append(
                    (message, progress)
                ),
            )
            result_path = output / "class_01" / "result.json"
            audit_path = output / "class_01" / "candidates" / "candidates.json"
            written = json.loads(result_path.read_text(encoding="utf-8"))

            self.assertEqual(len(result["pages"]), 2)
            self.assertEqual(result["pages"][1]["start_sec"], 10.0)
            self.assertEqual(written["video_id"], "class_01")
            self.assertTrue((output / "class_01" / "page_001.jpg").is_file())
            self.assertTrue((output / "class_01" / "page_002.jpg").is_file())
            with Image.open(output / "class_01" / "page_001.jpg") as screenshot:
                self.assertEqual(screenshot.size, (1024, 634))
            self.assertTrue(audit_path.is_file())
            self.assertEqual(progress_events[-1], ("处理完成", 1.0))

    @patch(
        "legacy_ffmpeg_scene_detector.pipeline.FFmpegTools",
        _FallbackFFmpegTools,
    )
    def test_adaptive_scene_threshold_falls_back_when_no_candidates(self) -> None:
        _FallbackFFmpegTools.thresholds = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "class.mp4"
            video.write_bytes(b"fake")
            result = VideoPageDetector(
                DetectorConfig(
                    scene_threshold=0.35,
                    scene_threshold_floor=0.05,
                    analysis_width=64,
                    analysis_height=36,
                )
            ).run(
                video,
                output_root=root / "output",
                video_id="adaptive",
            )
        self.assertEqual(
            _FallbackFFmpegTools.thresholds,
            [0.35, 0.175, 0.0875, 0.05],
        )
        self.assertEqual(result["analysis"]["scene_threshold_used"], 0.05)
        self.assertTrue(result["analysis"]["adaptive_fallback_applied"])
        self.assertEqual(len(result["pages"]), 2)
