import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from video_page_detector.batch_pipeline import BatchVideoPageDetector
from video_page_detector.config import DetectorConfig
from video_page_detector.ffmpeg_io import VideoMetadata
from video_page_detector.streaming_pipeline import (
    StreamingVideoPageDetector,
)


class _StreamingFakeFFmpegTools:
    analysis_chunks: list[float] = []

    def __init__(
        self,
        *,
        analysis_width: int,
        analysis_height: int,
        **_: object,
    ) -> None:
        self.width = analysis_width
        self.height = analysis_height
        self.states = [
            np.random.default_rng(seed).integers(
                0,
                256,
                size=(self.height, self.width, 3),
                dtype=np.uint8,
            )
            for seed in (101, 202, 303, 404)
        ]

    def probe(self, _: Path) -> VideoMetadata:
        return VideoMetadata(duration_sec=48.0, width=1280, height=720)

    def sample_window(
        self,
        _: Path,
        *,
        start_sec: float,
        duration_sec: float,
        fps: float,
    ) -> list[np.ndarray]:
        if fps == 0.5 and duration_sec > 10:
            self.__class__.analysis_chunks.append(start_sec)
        count = max(1, int(duration_sec * fps))
        frames = []
        for index in range(count):
            timestamp = start_sec + index / fps
            state_index = min(3, int(timestamp // 12.0))
            frames.append(self.states[state_index].copy())
        return frames


class StreamingTemporalPipelineTests(unittest.TestCase):
    @staticmethod
    def _config() -> DetectorConfig:
        return DetectorConfig(
            temporal_sample_interval_sec=2.0,
            temporal_confirmation_sec=6.0,
            temporal_analysis_width=64,
            temporal_analysis_height=36,
            temporal_chunk_duration_sec=18.0,
            auto_detect_screen_crop=False,
            min_page_duration_sec=4.0,
            screen_crop_left_ratio=0.0,
            screen_crop_top_ratio=0.0,
            screen_crop_right_ratio=0.0,
            screen_crop_bottom_ratio=0.0,
        )

    @patch(
        "video_page_detector.streaming_pipeline.FFmpegTools",
        _StreamingFakeFFmpegTools,
    )
    def test_detection_does_not_depend_on_stderr(self) -> None:
        class BrokenStderr:
            def write(self, *_: object) -> None:
                raise OSError(22, "Invalid argument")

            flush = write

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "class.mp4"
            video.write_bytes(b"fake")
            with patch("sys.stderr", BrokenStderr()):
                result = StreamingVideoPageDetector(self._config()).run(
                    video,
                    output_root=root / "streaming",
                    video_id="lesson",
                )
        self.assertTrue(result["pages"])

    @patch(
        "video_page_detector.streaming_pipeline.FFmpegTools",
        _StreamingFakeFFmpegTools,
    )
    @patch(
        "video_page_detector.batch_pipeline.FFmpegTools",
        _StreamingFakeFFmpegTools,
    )
    def test_streaming_matches_batch_and_emits_before_eof(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "class.mp4"
            video.write_bytes(b"fake")
            config = self._config()

            _StreamingFakeFFmpegTools.analysis_chunks = []
            batch = BatchVideoPageDetector(config).run(
                video,
                output_root=root / "batch",
                video_id="lesson",
            )

            _StreamingFakeFFmpegTools.analysis_chunks = []
            callback_chunk_counts: list[int] = []
            streaming = StreamingVideoPageDetector(config).run(
                video,
                output_root=root / "streaming",
                video_id="lesson",
                page_ready_callback=lambda _page, _completed, _total: (
                    callback_chunk_counts.append(
                        len(_StreamingFakeFFmpegTools.analysis_chunks)
                    )
                ),
            )

            self.assertEqual(len(streaming["pages"]), len(batch["pages"]))
            self.assertEqual(
                [
                    (page["start_sec"], page["end_sec"])
                    for page in streaming["pages"]
                ],
                [
                    (page["start_sec"], page["end_sec"])
                    for page in batch["pages"]
                ],
            )
            self.assertTrue(callback_chunk_counts)
            self.assertLess(
                callback_chunk_counts[0],
                len(_StreamingFakeFFmpegTools.analysis_chunks),
            )
            self.assertTrue(
                streaming["analysis"]["streaming_page_confirmation"]
            )
            self.assertIsNotNone(
                streaming["analysis"]["first_page_ready_after_sec"]
            )


if __name__ == "__main__":
    unittest.main()
