import json
import os
import tempfile
import threading
import time
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import patch

from video_page_detector.desktop_v2_worker import (
    delete_task_page,
    delete_task_result,
    detector_algorithm,
    discover_original_video,
    emit,
    emit_page,
    handle_command,
    llm_config,
    list_tasks,
    launch_pending_work,
    load_run_elapsed,
    load_task,
    page_speech_text,
    replay_active_events,
    run_task,
    run_retry_failed_pages,
    scene_threshold_detector_config,
    start_task,
    start_retry_failed_pages,
    transcription_config,
)


class DesktopV2WorkerTests(unittest.TestCase):
    def test_pending_tasks_launch_in_fifo_order(self) -> None:
        pending: deque[dict] = deque([
            {"_queue_id": "queue-1", "video_id": "one"},
            {"_queue_id": "queue-2", "video_id": "two"},
        ])
        with (
            patch("video_page_detector.desktop_v2_worker._active_thread", None),
            patch("video_page_detector.desktop_v2_worker._detection_busy", False),
            patch("video_page_detector.desktop_v2_worker._active_task_dirs", set()),
            patch("video_page_detector.desktop_v2_worker._pending_task_payloads", pending),
            patch("video_page_detector.desktop_v2_worker._pending_retry_payload", None),
            patch("video_page_detector.desktop_v2_worker.threading.Thread") as thread,
            patch("video_page_detector.desktop_v2_worker.emit") as emit,
        ):
            launch_pending_work()

        self.assertEqual(thread.call_args.kwargs["target"], run_task)
        self.assertEqual(thread.call_args.kwargs["args"][0]["video_id"], "one")
        thread.return_value.start.assert_called_once_with()
        emit.assert_called_once_with(
            "task.queue_updated", task_id="queue-2", queue_position=1
        )

    def test_new_task_is_queued_while_another_task_is_running(self) -> None:
        pending: deque[dict] = deque()
        with (
            patch("video_page_detector.desktop_v2_worker._detection_busy", True),
            patch("video_page_detector.desktop_v2_worker._pending_task_payloads", pending),
            patch("video_page_detector.desktop_v2_worker.time.time_ns", return_value=123),
            patch("video_page_detector.desktop_v2_worker.emit") as emit,
        ):
            start_task({"video_path": "two.mp4", "video_id": "two", "mode": "full"})

        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["_queue_id"], "queue-123")
        emit.assert_called_once_with(
            "task.queued",
            task_id="queue-123",
            video_id="two",
            video_path="two.mp4",
            queue_position=1,
            mode="full",
            include_llm=True,
        )

    def test_new_task_starts_while_previous_task_only_uses_cloud(self) -> None:
        with (
            patch("video_page_detector.desktop_v2_worker._detection_busy", False),
            patch(
                "video_page_detector.desktop_v2_worker._active_task_dirs",
                {Path("previous-task")},
            ),
            patch("video_page_detector.desktop_v2_worker.threading.Thread") as thread,
        ):
            start_task({"video_path": "next.mp4", "video_id": "next"})

        thread.return_value.start.assert_called_once_with()
        self.assertEqual(thread.call_args.kwargs["target"], run_task)

    def test_active_events_replay_after_frontend_refresh(self) -> None:
        with patch("video_page_detector.desktop_v2_worker._write_event") as write:
            emit("task.started", task_id="task-1")
            emit("page.updated", task_id="task-1", page={"page_id": 1})
            write.reset_mock()

            replay_active_events()

            self.assertEqual(
                [call.args[0]["type"] for call in write.call_args_list],
                ["task.started", "page.updated"],
            )
            emit("task.completed", task_id="task-1")

    def test_completed_page_delete_removes_and_renumbers_all_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run_dir = root / "lesson"
            evaluation_dir = run_dir / "llm_evaluation"
            evaluation_dir.mkdir(parents=True)
            pages = [
                {
                    "page_id": page_id,
                    "start_sec": page_id - 1,
                    "end_sec": page_id,
                    "screenshot_path": f"page_{page_id:03d}.jpg",
                }
                for page_id in (1, 2, 3)
            ]
            (run_dir / "result.json").write_text(
                json.dumps({"video_id": "lesson", "pages": pages}), encoding="utf-8"
            )
            (run_dir / "transcript.json").write_text(
                json.dumps({
                    "video_id": "lesson",
                    "pages": [{**page, "utterances": []} for page in pages],
                    "transcription": {"cloud_statistics": {"failed_page_count": 0}},
                }),
                encoding="utf-8",
            )
            (evaluation_dir / "llm_evaluation.json").write_text(
                json.dumps({
                    "video_id": "lesson",
                    "model": "test-model",
                    "prompt_version": "test",
                    "pages": [
                        {"page_id": page_id, "status": "scored", "score": 80}
                        for page_id in (1, 2, 3)
                    ],
                    "summary": {},
                }),
                encoding="utf-8",
            )
            (run_dir / "run_metadata.json").write_text(
                json.dumps({"status": "completed", "mode": "full", "include_llm": True}),
                encoding="utf-8",
            )

            task = delete_task_page(str(run_dir), 2, str(root))

            self.assertIsNotNone(task)
            self.assertEqual([page["page_id"] for page in task["pages"]], [1, 2])
            self.assertTrue(task["pages"][1]["screenshot_path"].endswith("page_003.jpg"))
            evaluation = json.loads(
                (evaluation_dir / "llm_evaluation.json").read_text(encoding="utf-8")
            )
            self.assertEqual(evaluation["summary"]["total_pages"], 2)

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
        self.assertEqual(emit.call_args.kwargs["algorithm_version"], "1.4.17")

    def test_detailed_evidence_setting_reaches_llm_config(self) -> None:
        self.assertTrue(llm_config({"include_evidence": True}).include_evidence)
        self.assertFalse(llm_config({"include_evidence": False}).include_evidence)

    def test_desktop_cloud_concurrency_settings_are_capped_at_ten(self) -> None:
        self.assertEqual(llm_config({"llm_concurrency": 10}).max_concurrency, 10)
        self.assertEqual(
            transcription_config({"asr_concurrency": 10}).mimo_max_concurrency,
            10,
        )

    def test_desktop_detector_algorithm_selection(self) -> None:
        self.assertEqual(detector_algorithm({}), "temporal")
        self.assertEqual(
            detector_algorithm({"detector_algorithm": "scene-threshold"}),
            "scene-threshold",
        )
        with self.assertRaisesRegex(ValueError, "不支持"):
            detector_algorithm({"detector_algorithm": "unknown"})
        scene_threshold_detector_config().validate()

    def test_local_asr_uses_selected_model_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            model_dir = Path(temp) / "faster-whisper-model"
            model_dir.mkdir()
            (model_dir / "model.bin").write_bytes(b"weights")

            config = transcription_config(
                {
                    "asr_engine": "faster-whisper",
                    "asr_model": str(model_dir),
                }
            )

            self.assertEqual(config.model, str(model_dir.resolve()))

    def test_local_asr_rejects_incomplete_model_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "model.bin"):
                transcription_config(
                    {
                        "asr_engine": "faster-whisper",
                        "asr_model": temp,
                    }
                )

    def test_discovers_legacy_task_video_without_file_picker(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            nested = Path(temp) / "course" / "videos"
            nested.mkdir(parents=True)
            video = nested / "lesson-42.mp4"
            video.write_bytes(b"video")

            discovered = discover_original_video(
                "lesson-42",
                search_roots=[Path(temp)],
            )

            self.assertEqual(discovered, video.resolve())

    def test_retry_is_queued_while_main_task_is_running(self) -> None:
        payload = {"task_id": "task-1", "page_ids": [2]}
        with (
            patch("video_page_detector.desktop_v2_worker._detection_busy", True),
            patch("video_page_detector.desktop_v2_worker._pending_retry_payload", None),
            patch("video_page_detector.desktop_v2_worker.emit") as emit,
        ):
            start_retry_failed_pages(payload)

        emit.assert_called_once_with(
            "task.retry_queued",
            task_id="task-1",
            page_ids=[2],
            message="当前主流程仍在处理，失败页重试已加入队列。",
        )

    def test_llm_retry_uses_configured_five_way_concurrency(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "retry-concurrency"
            evaluation_dir = run_dir / "llm_evaluation"
            evaluation_dir.mkdir(parents=True)
            pages = [
                {
                    "page_id": page_id,
                    "start_sec": page_id * 5,
                    "end_sec": (page_id + 1) * 5,
                    "screenshot_path": str(run_dir / f"page_{page_id:03d}.jpg"),
                    "speech_text": f"第{page_id}页讲话",
                    "utterances": [{"text": f"第{page_id}页讲话"}],
                }
                for page_id in range(1, 7)
            ]
            (run_dir / "result.json").write_text(
                json.dumps({"video_id": "retry-concurrency", "pages": pages}),
                encoding="utf-8",
            )
            (run_dir / "transcript.json").write_text(
                json.dumps({"video_id": "retry-concurrency", "pages": pages}),
                encoding="utf-8",
            )
            (evaluation_dir / "llm_evaluation.json").write_text(
                json.dumps(
                    {
                        "video_id": "retry-concurrency",
                        "pages": [
                            {
                                "page_id": page["page_id"],
                                "status": "failed",
                                "failure_stage": "llm",
                                "score": 0,
                                "reason": "timeout",
                            }
                            for page in pages
                        ],
                        "summary": {"failed_pages": 6, "complete": False},
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "status": "completed_with_errors",
                        "mode": "full",
                        "include_llm": True,
                    }
                ),
                encoding="utf-8",
            )
            counter_lock = threading.Lock()
            current = 0
            peak = 0

            def score_page(page, **_kwargs):
                nonlocal current, peak
                with counter_lock:
                    current += 1
                    peak = max(peak, current)
                time.sleep(0.08)
                with counter_lock:
                    current -= 1
                return {
                    "page_id": page["page_id"],
                    "start_sec": page["start_sec"],
                    "end_sec": page["end_sec"],
                    "status": "scored",
                    "score": 90,
                    "reason": "相关",
                }

            with (
                patch(
                    "video_page_detector.desktop_v2_worker.evaluate_page",
                    side_effect=score_page,
                ),
                patch("video_page_detector.desktop_v2_worker.emit") as emit,
            ):
                run_retry_failed_pages(
                    {
                        "task_id": str(run_dir),
                        "output_root": temp,
                        "page_ids": [1, 2, 3, 4, 5, 6],
                        "settings": {
                            "llm_concurrency": 5,
                            "include_evidence": False,
                        },
                        "llm_api_key": "test-key",
                        "llm_upload_consent": True,
                    }
                )

            retry_started = next(
                call for call in emit.call_args_list
                if call.args == ("task.retry_started",)
            )
            activity_values = [
                call.kwargs["active_cloud_requests"]
                for call in emit.call_args_list
                if call.args == ("cloud.activity",)
            ]
            self.assertEqual(retry_started.kwargs["llm_concurrency"], 5)
            self.assertEqual(peak, 5)
            self.assertEqual(max(activity_values), 5)

    def test_asr_retry_uses_configured_three_way_concurrency(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "retry-asr-concurrency"
            run_dir.mkdir()
            video_path = Path(temp) / "retry-asr-concurrency.mp4"
            video_path.write_bytes(b"video")
            pages = [
                {
                    "page_id": page_id,
                    "start_sec": page_id * 5,
                    "end_sec": (page_id + 1) * 5,
                    "speech_text": "",
                    "utterances": [],
                    "failure_stage": "asr",
                    "transcription_status": "failed",
                }
                for page_id in range(1, 5)
            ]
            (run_dir / "result.json").write_text(
                json.dumps({"video_id": "retry-asr-concurrency", "pages": pages}),
                encoding="utf-8",
            )
            (run_dir / "transcript.json").write_text(
                json.dumps(
                    {
                        "video_id": "retry-asr-concurrency",
                        "video_path": str(video_path),
                        "pages": pages,
                        "transcription": {
                            "cloud_statistics": {"failed_page_count": 4}
                        },
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "status": "completed_with_errors",
                        "mode": "full",
                        "include_llm": False,
                    }
                ),
                encoding="utf-8",
            )
            counter_lock = threading.Lock()
            current = 0
            peak = 0

            def transcribe(_video, retry_pages, **_kwargs):
                nonlocal current, peak
                with counter_lock:
                    current += 1
                    peak = max(peak, current)
                time.sleep(0.08)
                with counter_lock:
                    current -= 1
                page = dict(retry_pages[0])
                page.update(
                    {
                        "speech_text": "恢复成功",
                        "utterances": [{"text": "恢复成功"}],
                    }
                )
                return [page], {}

            with (
                patch(
                    "video_page_detector.desktop_v2_worker.transcribe_pages_with_mimo",
                    side_effect=transcribe,
                ),
                patch("video_page_detector.desktop_v2_worker.emit") as emit,
            ):
                run_retry_failed_pages(
                    {
                        "task_id": str(run_dir),
                        "output_root": temp,
                        "page_ids": [1, 2, 3, 4],
                        "settings": {
                            "asr_engine": "mimo-cloud",
                            "asr_concurrency": 3,
                            "include_llm": False,
                        },
                        "asr_api_key": "test-key",
                        "asr_upload_consent": True,
                    }
                )

            retry_started = next(
                call for call in emit.call_args_list
                if call.args == ("task.retry_started",)
            )
            activity_values = [
                call.kwargs["active_cloud_requests"]
                for call in emit.call_args_list
                if call.args == ("cloud.activity",)
            ]
            self.assertEqual(retry_started.kwargs["asr_concurrency"], 3)
            self.assertEqual(peak, 3)
            self.assertEqual(max(activity_values), 3)

    def test_mixed_asr_and_llm_retries_start_in_parallel(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "mixed-retry"
            run_dir.mkdir()
            evaluation_dir = run_dir / "llm_evaluation"
            evaluation_dir.mkdir()
            video_path = Path(temp) / "mixed-retry.mp4"
            video_path.write_bytes(b"video")
            pages = [
                {
                    "page_id": 1,
                    "start_sec": 0,
                    "end_sec": 5,
                    "screenshot_path": str(run_dir / "page_001.jpg"),
                    "speech_text": "",
                    "utterances": [],
                    "failure_stage": "asr",
                    "transcription_status": "failed",
                },
                {
                    "page_id": 2,
                    "start_sec": 5,
                    "end_sec": 10,
                    "screenshot_path": str(run_dir / "page_002.jpg"),
                    "speech_text": "第二页讲话",
                    "utterances": [
                        {"start_sec": 5, "end_sec": 10, "text": "第二页讲话"}
                    ],
                },
            ]
            (run_dir / "result.json").write_text(
                json.dumps({"video_id": "mixed-retry", "pages": pages}),
                encoding="utf-8",
            )
            (run_dir / "transcript.json").write_text(
                json.dumps(
                    {
                        "video_id": "mixed-retry",
                        "video_path": str(video_path),
                        "pages": pages,
                        "transcription": {
                            "cloud_statistics": {"failed_page_count": 1}
                        },
                    }
                ),
                encoding="utf-8",
            )
            (evaluation_dir / "llm_evaluation.json").write_text(
                json.dumps(
                    {
                        "video_id": "mixed-retry",
                        "pages": [
                            {
                                "page_id": 2,
                                "status": "failed",
                                "failure_stage": "llm",
                                "score": 0,
                                "reason": "timeout",
                            }
                        ],
                        "summary": {"failed_pages": 1, "complete": False},
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "status": "completed_with_errors",
                        "mode": "full",
                        "include_llm": True,
                    }
                ),
                encoding="utf-8",
            )
            asr_started = threading.Event()
            llm_started = threading.Event()
            release = threading.Event()

            def transcribe(_video, retry_pages, **_kwargs):
                asr_started.set()
                release.wait(timeout=3)
                page = dict(retry_pages[0])
                page.update(
                    {
                        "speech_text": "第一页恢复成功",
                        "utterances": [
                            {
                                "start_sec": 0,
                                "end_sec": 5,
                                "text": "第一页恢复成功",
                            }
                        ],
                    }
                )
                return [page], {}

            def score(page, **_kwargs):
                if int(page["page_id"]) == 2:
                    llm_started.set()
                release.wait(timeout=3)
                return {
                    "page_id": page["page_id"],
                    "start_sec": page["start_sec"],
                    "end_sec": page["end_sec"],
                    "status": "scored",
                    "score": 90,
                    "reason": "相关",
                }

            with (
                patch(
                    "video_page_detector.desktop_v2_worker.transcribe_pages_with_mimo",
                    side_effect=transcribe,
                ),
                patch(
                    "video_page_detector.desktop_v2_worker.evaluate_page",
                    side_effect=score,
                ),
                patch("video_page_detector.desktop_v2_worker.emit") as emit,
            ):
                retry_thread = threading.Thread(
                    target=run_retry_failed_pages,
                    args=(
                        {
                            "task_id": str(run_dir),
                            "output_root": temp,
                            "page_ids": [1, 2],
                            "settings": {
                                "asr_engine": "mimo-cloud",
                                "asr_concurrency": 5,
                                "llm_concurrency": 5,
                                "include_evidence": False,
                            },
                            "asr_api_key": "test-key",
                            "llm_api_key": "test-key",
                            "asr_upload_consent": True,
                            "llm_upload_consent": True,
                        },
                    ),
                )
                retry_thread.start()
                try:
                    self.assertTrue(asr_started.wait(timeout=1))
                    self.assertTrue(llm_started.wait(timeout=1))
                finally:
                    release.set()
                    retry_thread.join(timeout=3)

            self.assertFalse(retry_thread.is_alive())
            retry_failures = [
                (call.kwargs.get("error"), call.kwargs.get("traceback"))
                for call in emit.call_args_list
                if call.args == ("task.retry_failed",)
            ]
            self.assertTrue(
                any(
                    call.args == ("task.retry_completed",)
                    for call in emit.call_args_list
                ),
                retry_failures,
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
            self.assertEqual(task["progress"], 100)
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

    def test_active_task_reload_remains_running(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "lesson"
            run_dir.mkdir()
            (run_dir / "result.json").write_text(
                json.dumps(
                    {
                        "video_id": "lesson",
                        "video_duration_sec": 10.0,
                        "pages": [
                            {
                                "page_id": 1,
                                "start_sec": 0.0,
                                "end_sec": 10.0,
                                "screenshot_path": "page_001.jpg",
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (run_dir / "run_metadata.json").write_text(
                json.dumps({"status": "running", "mode": "full"}),
                encoding="utf-8",
            )
            with patch(
                "video_page_detector.desktop_v2_worker._active_task_dirs",
                {run_dir.resolve()},
            ):
                task = load_task(run_dir)

            self.assertIsNotNone(task)
            assert task is not None
            self.assertEqual(task["status"], "running")
            self.assertEqual(task["stage"], "正在处理")
            self.assertEqual(task["stage_progresses"]["ppt"], 100)

    def test_persisted_completion_wins_while_worker_thread_is_finalizing(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "lesson"
            evaluation_dir = run_dir / "llm_evaluation"
            evaluation_dir.mkdir(parents=True)
            (run_dir / "result.json").write_text(
                json.dumps(
                    {
                        "video_id": "lesson",
                        "pages": [{"page_id": 1, "start_sec": 0, "end_sec": 5}],
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "transcript.json").write_text(
                json.dumps({"pages": [{"page_id": 1, "speech_text": "讲解"}]}),
                encoding="utf-8",
            )
            (evaluation_dir / "llm_evaluation.json").write_text(
                json.dumps(
                    {
                        "summary": {"complete": True, "failed_pages": 0},
                        "pages": [{"page_id": 1, "status": "scored", "score": 90}],
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "mode": "full",
                        "include_llm": True,
                        "elapsed_sec": 12.5,
                    }
                ),
                encoding="utf-8",
            )
            with patch(
                "video_page_detector.desktop_v2_worker._active_task_dirs",
                {run_dir.resolve()},
            ):
                task = load_task(run_dir)

            assert task is not None
            self.assertEqual(task["status"], "completed")
            self.assertEqual(task["progress"], 100)
            self.assertEqual(
                task["stage_progresses"],
                {"ppt": 100, "voice": 100, "llm": 100, "report": 100},
            )
            self.assertIn("report", task["completed_stages"])

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
            self.assertEqual(task["pages"][0]["failure_stage"], "llm")

    def test_failed_asr_page_reloads_with_failure_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "asr-partial"
            run_dir.mkdir()
            (run_dir / "result.json").write_text(
                json.dumps(
                    {
                        "video_id": "asr-partial",
                        "pages": [{"page_id": 1, "start_sec": 0, "end_sec": 5}],
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "transcript.json").write_text(
                json.dumps(
                    {
                        "pages": [
                            {
                                "page_id": 1,
                                "start_sec": 0,
                                "end_sec": 5,
                                "transcription_status": "failed",
                                "failure_stage": "asr",
                                "reason": "ASR timeout",
                            }
                        ],
                        "transcription": {
                            "cloud_statistics": {"failed_page_count": 1}
                        },
                    }
                ),
                encoding="utf-8",
            )

            task = load_task(run_dir)

            assert task is not None
            self.assertEqual(task["status"], "completed_with_errors")
            self.assertEqual(task["pages"][0]["status"], "failed")
            self.assertEqual(task["pages"][0]["failure_stage"], "asr")

    def test_interrupted_task_without_transcript_reloads_as_retryable_asr(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "interrupted"
            run_dir.mkdir()
            video_path = Path(temp) / "lesson.mp4"
            video_path.write_bytes(b"video")
            (run_dir / "result.json").write_text(
                json.dumps(
                    {
                        "video_id": "interrupted",
                        "video_path": str(video_path),
                        "pages": [{"page_id": 1, "start_sec": 0, "end_sec": 5}],
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "status": "running",
                        "mode": "full",
                        "include_llm": True,
                    }
                ),
                encoding="utf-8",
            )

            task = load_task(run_dir)

            assert task is not None
            self.assertEqual(task["status"], "completed_with_errors")
            self.assertEqual(task["pages"][0]["status"], "failed")
            self.assertEqual(task["pages"][0]["failure_stage"], "asr")
            self.assertEqual(task["video_path"], str(video_path))

    def test_interrupted_task_rebuilds_transcript_for_asr_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "interrupted-retry"
            run_dir.mkdir()
            video_path = Path(temp) / "lesson.mp4"
            video_path.write_bytes(b"video")
            source_page = {"page_id": 1, "start_sec": 0, "end_sec": 5}
            (run_dir / "result.json").write_text(
                json.dumps({"video_id": "interrupted-retry", "pages": [source_page]}),
                encoding="utf-8",
            )
            (run_dir / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "status": "running",
                        "mode": "full",
                        "include_llm": False,
                    }
                ),
                encoding="utf-8",
            )
            recovered_page = {
                **source_page,
                "speech_text": "恢复成功",
                "utterances": [{"start_sec": 0, "end_sec": 2, "text": "恢复成功"}],
            }

            with (
                patch(
                    "video_page_detector.desktop_v2_worker.discover_original_video",
                    return_value=video_path,
                ) as discover_video,
                patch(
                    "video_page_detector.desktop_v2_worker.transcribe_pages_with_mimo",
                    return_value=([recovered_page], {}),
                ),
                patch("video_page_detector.desktop_v2_worker.emit") as emit,
            ):
                run_retry_failed_pages(
                    {
                        "task_id": str(run_dir),
                        "output_root": temp,
                        "page_ids": [1],
                        "settings": {"asr_engine": "mimo-cloud"},
                        "asr_api_key": "test-key",
                        "asr_upload_consent": True,
                        "llm_upload_consent": False,
                    }
                )

            discover_video.assert_called_once()
            transcript = json.loads(
                (run_dir / "transcript.json").read_text(encoding="utf-8")
            )
            self.assertEqual(transcript["pages"][0]["speech_text"], "恢复成功")
            self.assertNotIn("failure_stage", transcript["pages"][0])
            self.assertTrue(
                any(call.args == ("task.retry_completed",) for call in emit.call_args_list)
            )

    def test_deletes_task_result_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "delete-me"
            run_dir.mkdir()
            (run_dir / "result.json").write_text(
                json.dumps({"video_id": "delete-me", "pages": []}),
                encoding="utf-8",
            )

            deleted = delete_task_result(str(run_dir), temp)

            self.assertEqual(deleted, run_dir.resolve())
            self.assertFalse(run_dir.exists())

    def test_deletes_failed_task_with_stale_running_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "stale-running"
            run_dir.mkdir()
            (run_dir / "result.json").write_text(
                json.dumps(
                    {
                        "video_id": "stale-running",
                        "pages": [{"page_id": 1, "start_sec": 0, "end_sec": 5}],
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "run_metadata.json").write_text(
                json.dumps({"status": "running"}),
                encoding="utf-8",
            )

            delete_task_result(str(run_dir), temp)

            self.assertFalse(run_dir.exists())

    def test_refuses_to_delete_the_actively_running_task(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "actually-running"
            run_dir.mkdir()
            (run_dir / "result.json").write_text(
                json.dumps(
                    {
                        "video_id": "actually-running",
                        "pages": [{"page_id": 1, "start_sec": 0, "end_sec": 5}],
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch(
                    "video_page_detector.desktop_v2_worker._active_task_dirs",
                    {run_dir.resolve()},
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "不能删除"):
                    delete_task_result(str(run_dir), temp)

            self.assertTrue(run_dir.exists())

    def test_asr_failed_page_retry_continues_to_llm(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "retry-asr"
            evaluation_dir = run_dir / "llm_evaluation"
            evaluation_dir.mkdir(parents=True)
            video_path = run_dir / "lesson.mp4"
            video_path.write_bytes(b"video")
            detection_page = {
                "page_id": 1,
                "start_sec": 0,
                "end_sec": 5,
                "screenshot_path": "page.jpg",
            }
            (run_dir / "result.json").write_text(
                json.dumps({"video_id": "retry-asr", "pages": [detection_page]}),
                encoding="utf-8",
            )
            (run_dir / "transcript.json").write_text(
                json.dumps(
                    {
                        "video_id": "retry-asr",
                        "video_path": str(video_path),
                        "pages": [
                            {
                                **detection_page,
                                "speech_text": "",
                                "utterances": [],
                                "transcription_status": "failed",
                                "failure_stage": "asr",
                                "reason": "timeout",
                            }
                        ],
                        "transcription": {
                            "cloud_statistics": {"failed_page_count": 1}
                        },
                        "artifacts": {
                            "page_transcript_markdown": str(run_dir / "逐页语音文字.md")
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (evaluation_dir / "llm_evaluation.json").write_text(
                json.dumps(
                    {
                        "video_id": "retry-asr",
                        "summary": {"failed_pages": 1, "complete": False},
                        "pages": [
                            {
                                "page_id": 1,
                                "start_sec": 0,
                                "end_sec": 5,
                                "status": "failed",
                                "failure_stage": "asr",
                                "speech_relevance": 0,
                                "ppt_coverage": 0,
                                "evidence_consistency": 0,
                                "score": 0,
                                "level": "请求失败",
                                "reason": "ASR timeout",
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (run_dir / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "status": "completed_with_errors",
                        "elapsed_sec": 5,
                        "mode": "full",
                        "include_llm": True,
                    }
                ),
                encoding="utf-8",
            )
            asr_page = {
                **detection_page,
                "speech_text": "牛顿第二定律",
                "utterances": [
                    {"start_sec": 0, "end_sec": 4, "text": "牛顿第二定律"}
                ],
            }
            scored_page = {
                "page_id": 1,
                "start_sec": 0,
                "end_sec": 5,
                "status": "scored",
                "speech_relevance": 90,
                "ppt_coverage": 90,
                "evidence_consistency": 90,
                "score": 90,
                "level": "高度相关",
                "reason": "内容一致",
            }

            with (
                patch(
                    "video_page_detector.desktop_v2_worker.transcribe_pages_with_mimo",
                    return_value=([asr_page], {}),
                ) as retry_asr,
                patch(
                    "video_page_detector.desktop_v2_worker.evaluate_page",
                    return_value=scored_page,
                ) as retry_llm,
                patch("video_page_detector.desktop_v2_worker.emit") as emit,
            ):
                run_retry_failed_pages(
                    {
                        "task_id": str(run_dir),
                        "output_root": temp,
                        "page_ids": [1],
                        "settings": {
                            "asr_engine": "mimo-cloud",
                            "include_evidence": False,
                        },
                        "asr_api_key": "test-asr-key",
                        "llm_api_key": "test-llm-key",
                        "asr_upload_consent": True,
                        "llm_upload_consent": True,
                    }
                )

            retry_asr.assert_called_once()
            retry_llm.assert_called_once()
            transcript = json.loads((run_dir / "transcript.json").read_text(encoding="utf-8"))
            evaluation = json.loads(
                (evaluation_dir / "llm_evaluation.json").read_text(encoding="utf-8")
            )
            self.assertEqual(transcript["pages"][0]["speech_text"], "牛顿第二定律")
            self.assertNotIn("failure_stage", transcript["pages"][0])
            self.assertEqual(transcript["transcription"]["cloud_statistics"]["failed_page_count"], 0)
            self.assertEqual(evaluation["pages"][0]["status"], "scored")
            self.assertEqual(evaluation["summary"]["failed_pages"], 0)
            self.assertTrue(any(call.args == ("task.retry_completed",) for call in emit.call_args_list))

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
