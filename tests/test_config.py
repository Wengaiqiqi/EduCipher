import json
import tempfile
import unittest
from pathlib import Path

from video_page_detector.config import DetectorConfig


class DetectorConfigTests(unittest.TestCase):
    def test_default_configuration_is_valid(self) -> None:
        DetectorConfig().validate()

    def test_unknown_key_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown"):
            DetectorConfig.from_mapping({"not_a_setting": 1})

    def test_confirmation_must_cover_two_samples(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two samples"):
            DetectorConfig(
                temporal_sample_interval_sec=4.0,
                temporal_confirmation_sec=6.0,
            ).validate()

    def test_screen_crop_cannot_remove_entire_frame(self) -> None:
        with self.assertRaisesRegex(ValueError, "horizontal"):
            DetectorConfig(
                screen_crop_left_ratio=0.6,
                screen_crop_right_ratio=0.4,
            ).validate()

    def test_loads_json_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(
                json.dumps({"temporal_sample_interval_sec": 1.5}),
                encoding="utf-8",
            )
            config = DetectorConfig.from_file(path)
        self.assertEqual(config.temporal_sample_interval_sec, 1.5)
        self.assertEqual(config.grid_columns, 4)
