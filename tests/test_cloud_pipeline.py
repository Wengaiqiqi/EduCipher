import tempfile
import threading
import time
import unittest
from pathlib import Path

from video_page_detector.cloud_pipeline import CloudPagePipeline
from video_page_detector.llm_evaluation import LLMEvaluationConfig
from video_page_detector.transcription import TranscriptionConfig


class CloudPagePipelineTests(unittest.TestCase):
    def test_deleted_page_skips_queued_asr_and_is_not_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "lesson.mp4"
            video.write_bytes(b"video")
            first_started = threading.Event()
            release = threading.Event()
            requested: list[int] = []

            def asr_runner(page: dict) -> tuple[dict, dict]:
                requested.append(int(page["page_id"]))
                first_started.set()
                release.wait(timeout=2)
                return {**page, "speech_text": "讲话", "utterances": []}, {}

            pipeline = CloudPagePipeline(
                video_path=video,
                result_path=root / "result.json",
                output_dir=root,
                transcription_config=TranscriptionConfig(
                    engine="mimo-cloud",
                    mimo_max_concurrency=1,
                ),
                asr_api_key="",
                asr_runner=asr_runner,
            )
            pages = [
                {"page_id": page_id, "start_sec": page_id - 1, "end_sec": page_id}
                for page_id in (1, 2)
            ]
            pipeline.submit_page(pages[0], 1, 2)
            pipeline.submit_page(pages[1], 2, 2)
            self.assertTrue(first_started.wait(timeout=1))
            pipeline.delete_page(2)
            release.set()

            transcript, _ = pipeline.finish({"video_id": "lesson", "pages": pages})

            self.assertEqual(requested, [1])
            self.assertEqual([page["page_id"] for page in transcript["pages"]], [1])

    def test_combined_cloud_concurrency_is_capped_at_ten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "lesson.mp4"
            video.write_bytes(b"video")

            def asr_runner(page: dict) -> tuple[dict, dict]:
                return {**page, "utterances": []}, {}

            def llm_runner(page: dict) -> dict:
                return {"page_id": page["page_id"], "status": "no_speech", "score": 0}

            pipeline = CloudPagePipeline(
                video_path=video,
                result_path=root / "result.json",
                output_dir=root,
                transcription_config=TranscriptionConfig(
                    engine="mimo-cloud",
                    mimo_max_concurrency=10,
                ),
                asr_api_key="",
                llm_config=LLMEvaluationConfig(max_concurrency=10),
                llm_api_key="",
                asr_runner=asr_runner,
                llm_runner=llm_runner,
            )
            page = {"page_id": 1, "start_sec": 0.0, "end_sec": 1.0}
            pipeline.submit_page(page, 1, 1)
            transcript, _ = pipeline.finish({"video_id": "lesson", "pages": [page]})
            self.assertEqual(
                transcript["transcription"]["cloud_statistics"][
                    "combined_cloud_concurrency_limit"
                ],
                10,
            )

    def test_asr_concurrency_is_shared_across_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "lesson.mp4"
            video.write_bytes(b"video")
            release = threading.Event()
            two_started = threading.Event()
            lock = threading.Lock()
            current = 0
            peak = 0

            def asr_runner(page: dict) -> tuple[dict, dict]:
                nonlocal current, peak
                with lock:
                    current += 1
                    peak = max(peak, current)
                    if current == 2:
                        two_started.set()
                release.wait(timeout=3)
                with lock:
                    current -= 1
                return {**page, "utterances": []}, {}

            pipelines = [
                CloudPagePipeline(
                    video_path=video,
                    result_path=root / f"task-{index}" / "result.json",
                    output_dir=root / f"task-{index}",
                    transcription_config=TranscriptionConfig(
                        engine="mimo-cloud",
                        mimo_max_concurrency=2,
                    ),
                    asr_api_key="",
                    asr_runner=asr_runner,
                )
                for index in (1, 2)
            ]
            pages = [
                {"page_id": page_id, "start_sec": page_id - 1, "end_sec": page_id}
                for page_id in range(1, 4)
            ]
            for pipeline in pipelines:
                for page_id, page in enumerate(pages, start=1):
                    pipeline.submit_page(page, page_id, len(pages))

            self.assertTrue(two_started.wait(timeout=1))
            time.sleep(0.05)
            self.assertEqual(peak, 2)
            release.set()
            for pipeline in pipelines:
                pipeline.finish({"video_id": "lesson", "pages": pages})

            self.assertEqual(peak, 2)

    def test_cloud_activity_reports_real_active_and_peak_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "lesson.mp4"
            video.write_bytes(b"video")
            release = threading.Event()
            all_started = threading.Event()
            started = 0
            started_lock = threading.Lock()
            activity: list[tuple[int, int]] = []

            def asr_runner(page: dict) -> tuple[dict, dict]:
                nonlocal started
                with started_lock:
                    started += 1
                    if started == 3:
                        all_started.set()
                release.wait(timeout=3)
                return {**page, "utterances": []}, {}

            pipeline = CloudPagePipeline(
                video_path=video,
                result_path=root / "result.json",
                output_dir=root,
                transcription_config=TranscriptionConfig(
                    engine="mimo-cloud",
                    mimo_max_concurrency=3,
                ),
                asr_api_key="",
                asr_runner=asr_runner,
                cloud_activity_callback=lambda active, limit: activity.append(
                    (active, limit)
                ),
            )
            pages = [
                {"page_id": page_id, "start_sec": page_id - 1, "end_sec": page_id}
                for page_id in range(1, 4)
            ]
            for page_id, page in enumerate(pages, start=1):
                pipeline.submit_page(page, page_id, len(pages))

            self.assertTrue(all_started.wait(timeout=1))
            self.assertEqual(max(active for active, _ in activity), 3)
            self.assertTrue(all(limit == 3 for _, limit in activity))
            release.set()
            transcript, _ = pipeline.finish(
                {"video_id": "lesson", "pages": pages}
            )
            self.assertEqual(activity[-1][0], 0)
            self.assertEqual(
                transcript["transcription"]["cloud_statistics"][
                    "peak_combined_cloud_requests"
                ],
                3,
            )

    def test_page_activity_only_marks_requests_when_they_really_start(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "lesson.mp4"
            video.write_bytes(b"video")
            release = threading.Event()
            first_started = threading.Event()
            activity: list[tuple[int, str]] = []

            def asr_runner(page: dict) -> tuple[dict, dict]:
                first_started.set()
                release.wait(timeout=3)
                return {**page, "utterances": []}, {}

            pipeline = CloudPagePipeline(
                video_path=video,
                result_path=root / "result.json",
                output_dir=root,
                transcription_config=TranscriptionConfig(
                    engine="mimo-cloud",
                    mimo_max_concurrency=1,
                ),
                asr_api_key="",
                asr_runner=asr_runner,
                page_activity_callback=lambda page, status: activity.append(
                    (int(page["page_id"]), status)
                ),
            )
            pages = [
                {"page_id": page_id, "start_sec": page_id - 1, "end_sec": page_id}
                for page_id in range(1, 3)
            ]
            for page_id, page in enumerate(pages, start=1):
                pipeline.submit_page(page, page_id, len(pages))

            self.assertTrue(first_started.wait(timeout=1))
            self.assertEqual(activity, [(1, "transcribing")])
            release.set()
            pipeline.finish({"video_id": "lesson", "pages": pages})
            self.assertEqual(
                activity,
                [(1, "transcribing"), (2, "transcribing")],
            )

    def test_abort_does_not_wait_for_running_cloud_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "lesson.mp4"
            video.write_bytes(b"video")
            started = threading.Event()
            release = threading.Event()

            def slow_asr(page: dict) -> tuple[dict, dict]:
                started.set()
                release.wait(timeout=3)
                return {**page, "utterances": []}, {}

            pipeline = CloudPagePipeline(
                video_path=video,
                result_path=root / "result.json",
                output_dir=root,
                transcription_config=TranscriptionConfig(
                    engine="mimo-cloud",
                    mimo_max_concurrency=1,
                ),
                asr_api_key="",
                asr_runner=slow_asr,
            )
            pipeline.submit_page(
                {"page_id": 1, "start_sec": 0.0, "end_sec": 1.0},
                1,
                1,
            )
            self.assertTrue(started.wait(timeout=1))
            before = time.perf_counter()
            pipeline.abort()
            elapsed = time.perf_counter() - before
            release.set()
            self.assertLess(elapsed, 0.5)

    def test_asr_result_immediately_starts_page_llm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "lesson.mp4"
            video.write_bytes(b"video")
            llm_started = threading.Event()
            llm_pages: list[int] = []

            def asr_runner(page: dict) -> tuple[dict, dict]:
                page_id = int(page["page_id"])
                result = dict(page)
                result.update(
                    {
                        "speech_text": f"第{page_id}页讲话",
                        "utterances": [
                            {
                                "start_sec": page["start_sec"],
                                "end_sec": page["end_sec"],
                                "text": f"第{page_id}页讲话",
                            }
                        ],
                        "cloud_request_duration_sec": 0.01,
                        "cloud_audio_chunk_count": 1,
                    }
                )
                return result, {
                    "audio_chunk_request_count": 1,
                    "uploaded_audio_bytes": 100,
                }

            def llm_runner(page: dict) -> dict:
                page_id = int(page["page_id"])
                llm_pages.append(page_id)
                llm_started.set()
                return {
                    "page_id": page_id,
                    "start_sec": page["start_sec"],
                    "end_sec": page["end_sec"],
                    "status": "scored",
                    "speech_relevance": 90,
                    "ppt_coverage": 80,
                    "evidence_consistency": 100,
                    "score": 89,
                    "level": "明显相关",
                    "reason": "内容相关",
                }

            pipeline = CloudPagePipeline(
                video_path=video,
                result_path=root / "result.json",
                output_dir=root,
                transcription_config=TranscriptionConfig(
                    engine="mimo-cloud",
                    mimo_max_concurrency=2,
                ),
                asr_api_key="",
                llm_config=LLMEvaluationConfig(max_concurrency=2),
                llm_api_key="",
                asr_runner=asr_runner,
                llm_runner=llm_runner,
            )
            page1 = {
                "page_id": 1,
                "start_sec": 0.0,
                "end_sec": 10.0,
                "screenshot_path": str(root / "page_001.jpg"),
            }
            page2 = {
                "page_id": 2,
                "start_sec": 10.0,
                "end_sec": 20.0,
                "screenshot_path": str(root / "page_002.jpg"),
            }
            pipeline.submit_page(page1, 1, 2)
            self.assertTrue(llm_started.wait(timeout=2.0))
            self.assertEqual(llm_pages, [1])
            pipeline.submit_page(page2, 2, 2)

            transcript, evaluation = pipeline.finish(
                {
                    "video_id": "lesson",
                    "video_duration_sec": 20.0,
                    "pages": [page1, page2],
                }
            )

            self.assertEqual(len(transcript["pages"]), 2)
            self.assertTrue(
                transcript["transcription"]["cloud_statistics"][
                    "streaming_pipeline"
                ]
            )
            self.assertEqual(
                transcript["transcription"]["cloud_statistics"][
                    "audio_chunk_request_count"
                ],
                2,
            )
            self.assertEqual(
                transcript["transcription"]["cloud_statistics"][
                    "combined_cloud_concurrency_limit"
                ],
                4,
            )
            self.assertIsNotNone(evaluation)
            assert evaluation is not None
            self.assertEqual(evaluation["summary"]["scored_pages"], 2)
            self.assertEqual(llm_pages, [1, 2])
            self.assertTrue((root / "transcript.json").is_file())
            self.assertTrue(
                (root / "llm_evaluation" / "llm_evaluation.json").is_file()
            )

    def test_asr_failure_is_persisted_for_page_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "lesson.mp4"
            video.write_bytes(b"video")

            def failing_asr(_: dict) -> tuple[dict, dict]:
                raise RuntimeError("cloud unavailable")

            pipeline = CloudPagePipeline(
                video_path=video,
                result_path=root / "result.json",
                output_dir=root,
                transcription_config=TranscriptionConfig(
                    engine="mimo-cloud",
                    mimo_max_concurrency=1,
                ),
                asr_api_key="",
                llm_config=LLMEvaluationConfig(max_concurrency=1),
                llm_api_key="test-key",
                asr_runner=failing_asr,
            )
            page = {
                "page_id": 1,
                "start_sec": 0.0,
                "end_sec": 10.0,
                "screenshot_path": str(root / "page_001.jpg"),
            }
            pipeline.submit_page(page, 1, 1)
            transcript, evaluation = pipeline.finish(
                {
                    "video_id": "lesson",
                    "video_duration_sec": 10.0,
                    "pages": [page],
                }
            )
            self.assertIsNotNone(evaluation)
            assert evaluation is not None
            self.assertEqual(transcript["pages"][0]["failure_stage"], "asr")
            self.assertEqual(
                transcript["pages"][0]["transcription_status"], "failed"
            )
            self.assertIn("cloud unavailable", transcript["pages"][0]["reason"])
            self.assertEqual(
                transcript["transcription"]["cloud_statistics"][
                    "failed_page_count"
                ],
                1,
            )
            self.assertEqual(evaluation["pages"][0]["status"], "failed")
            self.assertEqual(evaluation["pages"][0]["failure_stage"], "asr")
            self.assertEqual(evaluation["summary"]["failed_pages"], 1)


if __name__ == "__main__":
    unittest.main()
