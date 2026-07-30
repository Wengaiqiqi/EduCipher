"""Integration tests for the current temporal pipeline."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from video_page_detector.config import DetectorConfig
from video_page_detector.ffmpeg_io import VideoMetadata
from video_page_detector.pipeline import VideoPageDetector


class _TemporalFakeFFmpegTools:
    def __init__(
        self,
        *,
        analysis_width: int,
        analysis_height: int,
        **_: object,
    ) -> None:
        self.width = analysis_width
        self.height = analysis_height
        self.first = np.random.default_rng(100).integers(
            0,
            256,
            size=(self.height, self.width, 3),
            dtype=np.uint8,
        )
        self.second = np.random.default_rng(200).integers(
            0,
            256,
            size=(self.height, self.width, 3),
            dtype=np.uint8,
        )

    def probe(self, _: Path) -> VideoMetadata:
        return VideoMetadata(duration_sec=24.0, width=1280, height=720)

    def sample_window(
        self,
        _: Path,
        *,
        start_sec: float,
        duration_sec: float,
        fps: float,
    ) -> list[np.ndarray]:
        count = max(1, int(duration_sec * fps))
        frames = []
        for index in range(count):
            timestamp = start_sec + index / fps
            frame = self.first if timestamp < 12.0 else self.second
            frames.append(frame.copy())
        return frames


class TemporalPipelineIntegrationTests(unittest.TestCase):
    @patch(
        "video_page_detector.streaming_pipeline.FFmpegTools",
        _TemporalFakeFFmpegTools,
    )
    def test_temporal_mode_writes_page_intervals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "class.mp4"
            video.write_bytes(b"fake")
            config = DetectorConfig(
                temporal_sample_interval_sec=2.0,
                temporal_confirmation_sec=6.0,
                temporal_analysis_width=64,
                temporal_analysis_height=36,
                temporal_chunk_duration_sec=30.0,
                auto_detect_screen_crop=False,
                min_page_duration_sec=4.0,
                screen_crop_left_ratio=0.0,
                screen_crop_top_ratio=0.0,
                screen_crop_right_ratio=0.0,
                screen_crop_bottom_ratio=0.0,
            )
            ready_pages: list[tuple[int, int, int]] = []

            def page_ready(
                page: dict[str, object],
                completed: int,
                total: int,
            ) -> None:
                self.assertTrue(
                    Path(str(page["screenshot_path"])).is_file()
                )
                ready_pages.append(
                    (int(page["page_id"]), completed, total)
                )

            result = VideoPageDetector(config).run(
                video,
                output_root=root / "output",
                video_id="temporal",
                page_ready_callback=page_ready,
            )
            output_dir = root / "output" / "temporal"
            self.assertEqual(result["analysis"]["mode"], "temporal")
            self.assertEqual(len(result["pages"]), 2)
            self.assertLess(result["pages"][0]["end_sec"], 16.0)
            self.assertTrue((output_dir / "page_001.jpg").is_file())
            self.assertTrue((output_dir / "page_002.jpg").is_file())
            self.assertTrue((output_dir / "temporal" / "segments.json").is_file())
            with Image.open(output_dir / "page_001.jpg") as screenshot:
                self.assertEqual(screenshot.size, (1280, 720))
            self.assertEqual(
                result["analysis"]["source_resolution"],
                {"width": 1280, "height": 720},
            )
            self.assertEqual(
                result["analysis"]["screenshot_resolution"],
                {"width": 1280, "height": 720},
            )
            self.assertEqual(ready_pages, [(1, 1, 2), (2, 2, 2)])
