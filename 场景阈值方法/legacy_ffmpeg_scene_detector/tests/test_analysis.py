import unittest

import numpy as np

from legacy_ffmpeg_scene_detector.analysis import (
    classify_change,
    crop_frame,
    detect_screen_crop_ratios,
    select_stable_frame,
)
from legacy_ffmpeg_scene_detector.hashing import (
    compare_grid,
    hamming_distance,
    phash,
)


class HashingTests(unittest.TestCase):
    def test_identical_images_have_zero_distance(self) -> None:
        image = np.random.default_rng(1).integers(
            0, 256, size=(120, 160, 3), dtype=np.uint8
        )
        self.assertEqual(hamming_distance(phash(image), phash(image.copy())), 0)

    def test_grid_detects_local_and_global_changes(self) -> None:
        rng = np.random.default_rng(2)
        before = rng.integers(0, 256, size=(120, 160, 3), dtype=np.uint8)
        local = before.copy()
        local[:40, :40] = rng.integers(
            0, 256, size=(40, 40, 3), dtype=np.uint8
        )
        global_change = rng.integers(
            0, 256, size=(120, 160, 3), dtype=np.uint8
        )
        local_ratio, _, _ = compare_grid(
            before,
            local,
            rows=3,
            columns=4,
            changed_distance=10,
        )
        global_ratio, _, _ = compare_grid(
            before,
            global_change,
            rows=3,
            columns=4,
            changed_distance=10,
        )
        self.assertLess(local_ratio, global_ratio)
        self.assertLessEqual(local_ratio, 0.25)
        self.assertGreaterEqual(global_ratio, 0.75)


class ClassificationTests(unittest.TestCase):
    def test_detects_large_screen_rectangle(self) -> None:
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        frame[10:90, 20:180] = 220
        ratios = detect_screen_crop_ratios(
            [frame, frame.copy(), frame.copy()],
            fallback=(0.1, 0.1, 0.1, 0.1),
        )
        self.assertLessEqual(ratios[0], 0.05)
        self.assertLessEqual(ratios[2], 0.05)
        self.assertLessEqual(ratios[1], 0.05)
        self.assertLessEqual(ratios[3], 0.06)

    def test_crops_frame_using_normalized_margins(self) -> None:
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        cropped = crop_frame(
            frame,
            left_ratio=0.1,
            top_ratio=0.2,
            right_ratio=0.15,
            bottom_ratio=0.1,
        )
        self.assertEqual(cropped.shape, (70, 150, 3))

    def test_change_levels(self) -> None:
        self.assertEqual(
            classify_change(0.8, low_threshold=0.25, high_threshold=0.65),
            "high",
        )
        self.assertEqual(
            classify_change(0.5, low_threshold=0.25, high_threshold=0.65),
            "low",
        )
        self.assertIsNone(
            classify_change(0.25, low_threshold=0.25, high_threshold=0.65)
        )

    def test_stable_frame_returns_first_frame_below_threshold(self) -> None:
        frames = [
            np.full((8, 8, 3), 0, dtype=np.uint8),
            np.full((8, 8, 3), 100, dtype=np.uint8),
            np.full((8, 8, 3), 102, dtype=np.uint8),
            np.full((8, 8, 3), 102, dtype=np.uint8),
        ]
        selected = select_stable_frame(frames, first_index=1, threshold=0.02)
        self.assertTrue(selected.stable)
        self.assertEqual(selected.index, 2)

    def test_stable_frame_uses_least_difference_fallback(self) -> None:
        frames = [
            np.full((8, 8, 3), 0, dtype=np.uint8),
            np.full((8, 8, 3), 100, dtype=np.uint8),
            np.full((8, 8, 3), 130, dtype=np.uint8),
            np.full((8, 8, 3), 150, dtype=np.uint8),
        ]
        selected = select_stable_frame(frames, first_index=1, threshold=0.01)
        self.assertFalse(selected.stable)
        self.assertEqual(selected.index, 3)
