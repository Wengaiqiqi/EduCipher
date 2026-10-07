import unittest

from legacy_ffmpeg_scene_detector.config import DetectorConfig


class DetectorConfigTests(unittest.TestCase):
    def test_default_configuration_is_valid(self) -> None:
        DetectorConfig().validate()

    def test_invalid_threshold_order_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "low < high"):
            DetectorConfig(
                low_change_ratio=0.8,
                high_change_ratio=0.5,
            ).validate()

    def test_scene_threshold_floor_cannot_exceed_requested_threshold(self) -> None:
        with self.assertRaisesRegex(ValueError, "scene_threshold_floor"):
            DetectorConfig(
                scene_threshold=0.04,
                scene_threshold_floor=0.05,
            ).validate()

    def test_screen_crop_cannot_remove_entire_frame(self) -> None:
        with self.assertRaisesRegex(ValueError, "horizontal"):
            DetectorConfig(
                screen_crop_left_ratio=0.6,
                screen_crop_right_ratio=0.4,
            ).validate()
