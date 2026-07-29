import unittest

import numpy as np

from legacy_ffmpeg_scene_detector.models import WorkingPage
from legacy_ffmpeg_scene_detector.postprocess import merge_intervals, merge_short_pages


def _page(start: float) -> WorkingPage:
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    signature = np.zeros(64, dtype=bool)
    return WorkingPage(start, frame, signature, "high")


class PostprocessingTests(unittest.TestCase):
    def test_short_middle_page_merges_into_previous(self) -> None:
        pages = [_page(0.0), _page(10.0), _page(13.0), _page(30.0)]
        result = merge_short_pages(
            pages,
            video_duration=40.0,
            min_duration=5.0,
        )
        self.assertEqual([page.start_sec for page in result], [0.0, 13.0, 30.0])

    def test_short_first_page_extends_next_to_zero(self) -> None:
        pages = [_page(0.0), _page(3.0), _page(10.0)]
        result = merge_short_pages(
            pages,
            video_duration=20.0,
            min_duration=5.0,
        )
        self.assertEqual([page.start_sec for page in result], [0.0, 10.0])

    def test_unstable_windows_merge_and_filter(self) -> None:
        result = merge_intervals(
            [(10, 20), (23, 35), (37, 45), (100, 110)],
            max_gap=5,
            min_duration=30,
        )
        self.assertEqual(result, [(10.0, 45.0)])
