import unittest

from legacy_ffmpeg_scene_detector.config import DetectorConfig
from legacy_ffmpeg_scene_detector.gui import (
    ALL_FIELDS,
    config_from_gui_values,
    format_seconds,
)


class LegacyGuiHelperTests(unittest.TestCase):
    def test_formats_video_timestamps(self) -> None:
        self.assertEqual(format_seconds(65.125), "01:05.125")
        self.assertEqual(format_seconds(3661.5), "01:01:01.500")

    def test_builds_valid_config_from_gui_values(self) -> None:
        base = DetectorConfig()
        values = {
            key: str(getattr(base, key))
            for key, _, _, _ in ALL_FIELDS
        }
        values["scene_threshold"] = "0.12"
        config = config_from_gui_values(
            base,
            values,
            adaptive_scene_detection=False,
            auto_detect_screen_crop=True,
            crop_output_screenshots=True,
            ffmpeg_path=" C:/tools/ffmpeg.exe ",
        )
        self.assertEqual(config.scene_threshold, 0.12)
        self.assertFalse(config.adaptive_scene_detection)
        self.assertEqual(config.ffmpeg_path, "C:/tools/ffmpeg.exe")
        self.assertIsNone(config.ffprobe_path)

    def test_rejects_invalid_numeric_value(self) -> None:
        base = DetectorConfig()
        values = {
            key: str(getattr(base, key))
            for key, _, _, _ in ALL_FIELDS
        }
        values["high_change_ratio"] = "not-a-number"
        with self.assertRaisesRegex(ValueError, "格式不正确"):
            config_from_gui_values(
                base,
                values,
                adaptive_scene_detection=True,
                auto_detect_screen_crop=True,
                crop_output_screenshots=True,
            )
