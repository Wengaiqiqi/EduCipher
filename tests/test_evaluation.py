import json
import tempfile
import unittest
from pathlib import Path

from video_page_detector.evaluation import evaluate_results


class EvaluationTests(unittest.TestCase):
    def test_computes_acceptance_metrics(self) -> None:
        predicted = {
            "video_duration_sec": 3600,
            "processing_duration_sec": 200,
            "pages": [
                {"start_sec": 0},
                {"start_sec": 10.8},
                {"start_sec": 20.5},
                {"start_sec": 40},
            ],
        }
        truth = {
            "pages": [
                {"start_sec": 0},
                {"start_sec": 10},
                {"start_sec": 20},
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            predicted_path = Path(directory) / "predicted.json"
            truth_path = Path(directory) / "truth.json"
            predicted_path.write_text(json.dumps(predicted), encoding="utf-8")
            truth_path.write_text(json.dumps(truth), encoding="utf-8")
            result = evaluate_results(predicted_path, truth_path)
        self.assertEqual(result["recall"], 1.0)
        self.assertAlmostEqual(result["false_positive_rate"], 1 / 3, places=4)
        self.assertTrue(result["acceptance"]["timestamp_error_at_most_2_sec"])
        self.assertTrue(
            result["acceptance"]["processing_time_within_5_min_per_hour"]
        )
