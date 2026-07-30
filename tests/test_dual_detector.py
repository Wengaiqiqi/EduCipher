import unittest

from video_page_detector.dual_detector import (
    SceneHint,
    analyze_reuse,
    filter_short_scene_intervals,
    parse_scene_timestamp,
)


class DualDetectorTests(unittest.TestCase):
    def test_parse_scene_timestamp(self) -> None:
        line = "showinfo n: 1 pts: 100 pts_time:38.75 pos:0"
        self.assertEqual(parse_scene_timestamp(line), 38.75)
        self.assertIsNone(parse_scene_timestamp("unrelated"))

    def test_analyze_reuse_detects_extra_hint_inside_page(self) -> None:
        pages = [
            {"page_id": 1, "start_sec": 0.0, "end_sec": 10.0},
            {"page_id": 2, "start_sec": 10.0, "end_sec": 20.0},
            {"page_id": 3, "start_sec": 20.0, "end_sec": 30.0},
        ]
        hints = [
            SceneHint(timestamp_sec=10.2, emitted_after_sec=1.0),
            SceneHint(timestamp_sec=15.0, emitted_after_sec=1.5),
            SceneHint(timestamp_sec=19.8, emitted_after_sec=2.0),
        ]
        result = analyze_reuse(
            pages,
            hints,
            video_duration_sec=30.0,
            tolerance_sec=0.5,
        )
        self.assertEqual(result["matched_boundary_count"], 2)
        self.assertEqual(result["reusable_page_count"], 2)
        self.assertEqual(result["reprocess_page_count"], 1)
        decisions = result["page_decisions"]
        self.assertTrue(decisions[0]["reusable"])
        self.assertFalse(decisions[1]["reusable"])
        self.assertTrue(decisions[2]["reusable"])

    def test_filter_short_scene_intervals_discards_intro_flash(self) -> None:
        hints = [
            SceneHint(timestamp_sec=3.3, emitted_after_sec=0.2),
            SceneHint(timestamp_sec=38.7, emitted_after_sec=1.2),
            SceneHint(timestamp_sec=73.4, emitted_after_sec=2.2),
        ]
        filtered = filter_short_scene_intervals(
            hints,
            min_interval_sec=5.0,
        )
        self.assertEqual(
            [round(item.timestamp_sec, 1) for item in filtered],
            [38.7, 73.4],
        )


if __name__ == "__main__":
    unittest.main()
