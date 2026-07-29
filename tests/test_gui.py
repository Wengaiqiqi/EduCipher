import unittest

from video_page_detector.config import DetectorConfig
from video_page_detector.gui import (
    PARAMETER_FIELDS,
    config_from_gui_values,
    format_seconds,
)


class GuiHelperTests(unittest.TestCase):
    def test_formats_video_timestamps(self) -> None:
        self.assertEqual(format_seconds(65.125), "01:05.125")
        self.assertEqual(format_seconds(3661.5), "01:01:01.500")

    def test_builds_valid_config_from_text_fields(self) -> None:
        base = DetectorConfig()
        values = {
            key: str(getattr(base, key))
            for key, _, _, _ in PARAMETER_FIELDS
        }
        values["temporal_sample_interval_sec"] = "1.5"
        config = config_from_gui_values(
            base,
            values,
            ffmpeg_path=" C:/tools/ffmpeg.exe ",
        )
        self.assertEqual(config.temporal_sample_interval_sec, 1.5)
        self.assertEqual(config.ffmpeg_path, "C:/tools/ffmpeg.exe")
        self.assertIsNone(config.ffprobe_path)

    def test_rejects_invalid_gui_value(self) -> None:
        base = DetectorConfig()
        values = {
            key: str(getattr(base, key))
            for key, _, _, _ in PARAMETER_FIELDS
        }
        values["temporal_changed_block_ratio"] = "not-a-number"
        with self.assertRaisesRegex(ValueError, "格式不正确"):
            config_from_gui_values(base, values)
