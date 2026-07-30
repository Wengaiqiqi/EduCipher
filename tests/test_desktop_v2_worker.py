import json
import tempfile
import unittest
from pathlib import Path

from video_page_detector.desktop_v2_worker import (
    list_tasks,
    load_task,
    page_speech_text,
)


class DesktopV2WorkerTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
