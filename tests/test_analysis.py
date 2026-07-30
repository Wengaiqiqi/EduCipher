import unittest

import numpy as np

from video_page_detector.analysis import crop_frame, detect_screen_crop_ratios
from video_page_detector.hashing import hamming_distance, phash


class HashingTests(unittest.TestCase):
    def test_identical_images_have_zero_distance(self) -> None:
        image = np.random.default_rng(1).integers(
            0, 256, size=(120, 160, 3), dtype=np.uint8
        )
        self.assertEqual(hamming_distance(phash(image), phash(image.copy())), 0)


class ScreenAnalysisTests(unittest.TestCase):
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
