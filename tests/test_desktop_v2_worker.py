import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from video_page_detector.desktop_v2_worker import (
    emit_page,
    handle_command,
    llm_config,
    list_tasks,
    load_run_elapsed,
    load_task,
    page_speech_text,
    transcription_config,
)


class DesktopV2WorkerTests(unittest.TestCase):
    def test_page_update_emits_task_id_at_event_top_level(self) -> None:
        with patch("video_page_detector.desktop_v2_worker.emit") as emit:
            emit_page({"page_id": 1}, "detected", task_id="task-1")
        emit.assert_called_once()
        self.assertEqual(emit.call_args.args, ("page.updated",))
        self.assertEqual(emit.call_args.kwargs["task_id"], "task-1")
        self.assertNotIn("task_id", emit.call_args.kwargs["page"])

    def test_ping_replays_worker_ready_event(self) -> None:
        with patch("video_page_detector.desktop_v2_worker.emit") as emit:
            handle_command({"action": "ping"})
        self.assertEqual(emit.call_args.args, ("worker.ready",))
        self.assertEqual(emit.call_args.kwargs["algorithm_version"], "1.4.1")

    def test_detailed_evidence_setting_reaches_llm_config(self) -> None:
        self.assertTrue(llm_config({"include_evidence": True}).include_evidence)
        self.assertFalse(llm_config({"include_evidence": False}).include_evidence)

    def test_desktop_cloud_concurrency_settings_are_capped_at_ten(self) -> None:
        self.assertEqual(llm_config({"llm_concurrency": 10}).max_concurrency, 10)
        self.assertEqual(
            transcription_config({"asr_concurrency": 10}).mimo_max_concurrency,
            10,
        )

    def test_combines_utterance_text(self) -> None:
        text = page_speech_text(
            {
                "utterances": [
                    {"text": "第一句"},
                    {"text": "第二句"},
                ]
            }
        )
        self.assertEqual(text, "第一句\n第二句")

    def test_loads_existing_task_for_desktop(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "lesson"
            evaluation_dir = run_dir / "llm_evaluation"
            evaluation_dir.mkdir(parents=True)
            (run_dir / "result.json").write_text(
                json.dumps(
                    {
                        "video_id": "lesson",
                        "pages": [
                            {
                                "page_id": 1,
                                "start_sec": 0,
                                "end_sec": 10,
                                "screenshot_path": "page.jpg",
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (run_dir / "transcript.json").write_text(
                json.dumps(
                    {
                        "pages": [
                            {
                                "page_id": 1,
                                "utterances": [{"text": "牛顿第二定律"}],
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (evaluation_dir / "llm_evaluation.json").write_text(
                json.dumps(
                    {
                        "model": "mimo",
                        "summary": {"strict_average_score": 92},
                        "pages": [
                            {
                                "page_id": 1,
                                "status": "scored",
                                "score": 92,
                                "reason": "高度相关",
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            task = load_task(run_dir)

            self.assertIsNotNone(task)
            assert task is not None
            self.assertEqual(task["status"], "completed")
            self.assertEqual(task["pages"][0]["speech_text"], "牛顿第二定律")
            self.assertEqual(task["pages"][0]["score"], 92)
            self.assertEqual(list_tasks(temp)[0]["video_id"], "lesson")

    def test_detect_only_task_reloads_as_completed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "detect-only"
            run_dir.mkdir()
            (run_dir / "result.json").write_text(
                json.dumps(
                    {
                        "video_id": "detect-only",
                        "pages": [{"page_id": 1, "start_sec": 0, "end_sec": 5}],
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "mode": "detect",
                        "include_llm": False,
                        "elapsed_sec": 3.5,
                    }
                ),
                encoding="utf-8",
            )

            task = load_task(run_dir)

            assert task is not None
            self.assertEqual(task["status"], "completed")
            self.assertEqual(task["progress"], 100)
            self.assertEqual(task["mode"], "detect")
            self.assertFalse(task["include_llm"])

    def test_failed_page_evaluation_reloads_as_completed_with_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "partial"
            evaluation_dir = run_dir / "llm_evaluation"
            evaluation_dir.mkdir(parents=True)
            (run_dir / "result.json").write_text(
                json.dumps(
                    {
                        "video_id": "partial",
                        "pages": [{"page_id": 1, "start_sec": 0, "end_sec": 5}],
                    }
                ),
                encoding="utf-8",
            )
            (evaluation_dir / "llm_evaluation.json").write_text(
                json.dumps(
                    {
                        "summary": {"failed_pages": 1, "complete": False},
                        "pages": [{"page_id": 1, "status": "failed", "score": 0}],
                    }
                ),
                encoding="utf-8",
            )

            task = load_task(run_dir)

            assert task is not None
            self.assertEqual(task["status"], "completed_with_errors")
            self.assertEqual(task["pages"][0]["status"], "failed")

    def test_loads_detailed_evidence_and_infers_legacy_setting(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "lesson"
            evaluation_dir = run_dir / "llm_evaluation"
            evaluation_dir.mkdir(parents=True)
            (run_dir / "result.json").write_text(
                json.dumps(
                    {
                        "video_id": "lesson",
                        "processing_duration_sec": 10,
                        "pages": [
                            {
                                "page_id": 1,
                                "start_sec": 0,
                                "end_sec": 10,
                                "screenshot_path": "page.jpg",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (evaluation_dir / "llm_evaluation.json").write_text(
                json.dumps(
                    {
                        "pages": [
                            {
                                "page_id": 1,
                                "status": "scored",
                                "score": 90,
                                "matched_evidence": [
                                    {"ppt": "F=ma", "speech": "牛顿第二定律"}
                                ],
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            task = load_task(run_dir)

            assert task is not None
            self.assertTrue(task["include_evidence"])
            self.assertEqual(
                task["pages"][0]["evidence"],
                [{"ppt": "F=ma", "speech": "牛顿第二定律"}],
            )

    def test_reconstructs_elapsed_for_legacy_perf_counter_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "lesson"
            evaluation_dir = run_dir / "llm_evaluation"
            evaluation_dir.mkdir(parents=True)
            result_path = run_dir / "result.json"
            evaluation_path = evaluation_dir / "llm_evaluation.json"
            result_path.write_text(
                json.dumps({"processing_duration_sec": 10, "pages": []}),
                encoding="utf-8",
            )
            evaluation_path.write_text(json.dumps({"pages": []}), encoding="utf-8")
            (run_dir / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "started_at": 258543.2,
                        "elapsed_sec": 1785331204,
                    }
                ),
                encoding="utf-8",
            )
            base = time.time() - 60
            os.utime(result_path, (base + 10, base + 10))
            os.utime(evaluation_path, (base + 30, base + 30))

            self.assertAlmostEqual(load_run_elapsed(run_dir) or 0, 30, places=2)


if __name__ == "__main__":
    unittest.main()
