import unittest
from pathlib import Path

from video_page_detector.desktop_app import (
    build_workflow_paths,
    combine_page_rows,
    detector_config_for_preset,
    sanitize_video_id,
)


class DesktopAppHelperTests(unittest.TestCase):
    def test_builds_expected_workflow_paths(self) -> None:
        paths = build_workflow_paths(Path("output"), "lesson")
        self.assertEqual(paths.run_dir, (Path("output") / "lesson").resolve())
        self.assertEqual(paths.result_json, paths.run_dir / "result.json")
        self.assertEqual(
            paths.transcript_json,
            paths.run_dir / "transcript.json",
        )
        self.assertEqual(
            paths.evaluation_dir,
            paths.run_dir / "llm_evaluation",
        )

    def test_sanitizes_video_id(self) -> None:
        self.assertEqual(sanitize_video_id("", "lesson"), "lesson")
        with self.assertRaises(ValueError):
            sanitize_video_id("bad/name", "lesson")

    def test_fast_preset_reduces_sampling_rate(self) -> None:
        precise = detector_config_for_preset("智能精准（推荐）")
        fast = detector_config_for_preset("快速预览")
        self.assertGreater(
            fast.temporal_sample_interval_sec,
            precise.temporal_sample_interval_sec,
        )
        self.assertLess(
            fast.temporal_refinement_fps,
            precise.temporal_refinement_fps,
        )

    def test_combines_detection_transcript_and_scores(self) -> None:
        detection = {
            "pages": [
                {
                    "page_id": 1,
                    "start_sec": 0,
                    "end_sec": 10,
                    "confidence": "high",
                    "screenshot_path": "page.jpg",
                }
            ]
        }
        transcript = {
            "pages": [
                {
                    "page_id": 1,
                    "start_sec": 0,
                    "end_sec": 10,
                    "utterances": [{"text": "hello"}],
                }
            ]
        }
        evaluation = {
            "pages": [
                {
                    "page_id": 1,
                    "status": "scored",
                    "score": 88,
                    "level": "明显相关",
                }
            ]
        }
        rows = combine_page_rows(detection, transcript, evaluation)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["utterance_count"], 1)
        self.assertEqual(rows[0]["score"], 88)
        self.assertEqual(rows[0]["confidence"], "high")


if __name__ == "__main__":
    unittest.main()
