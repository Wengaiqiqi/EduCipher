from __future__ import annotations

import json
import os
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from video_page_detector.cloud_pipeline import CloudPagePipeline  # noqa: E402
from video_page_detector.config import DetectorConfig  # noqa: E402
from video_page_detector.llm_evaluation import LLMEvaluationConfig  # noqa: E402
from video_page_detector.streaming_pipeline import (  # noqa: E402
    StreamingVideoPageDetector,
)
from video_page_detector.transcription import TranscriptionConfig  # noqa: E402


def _read_user_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if value:
        return value
    if sys.platform != "win32":
        return ""
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            stored, _ = winreg.QueryValueEx(key, name)
    except OSError:
        return ""
    return str(stored).strip()


class TimedCloudPagePipeline(CloudPagePipeline):
    def __init__(self, *args: Any, clock_started_at: float, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.clock_started_at = clock_started_at
        self.first_page_submitted_after_sec: float | None = None
        self.first_asr_completed_after_sec: float | None = None
        self.first_llm_completed_after_sec: float | None = None
        self._timing_lock = threading.Lock()

    def _elapsed(self) -> float:
        return round(time.perf_counter() - self.clock_started_at, 3)

    def submit_page(self, page: dict, completed: int, total: int) -> None:
        with self._timing_lock:
            if self.first_page_submitted_after_sec is None:
                self.first_page_submitted_after_sec = self._elapsed()
        super().submit_page(page, completed, total)

    def _after_asr(self, page_id: int, future: Any) -> None:
        with self._timing_lock:
            if (
                self.first_asr_completed_after_sec is None
                and future.exception() is None
            ):
                self.first_asr_completed_after_sec = self._elapsed()
        super()._after_asr(page_id, future)

    def _after_llm(self, page_id: int, future: Any) -> None:
        with self._timing_lock:
            if (
                self.first_llm_completed_after_sec is None
                and future.exception() is None
            ):
                self.first_llm_completed_after_sec = self._elapsed()
        super()._after_llm(page_id, future)


def main() -> int:
    video = Path(r"E:\leeson\test_vedio\444.mp4")
    output_root = PROJECT_ROOT / "diagnostics" / "streaming_temporal_fullchain_444"
    video_id = "444_streaming_temporal_fullchain"
    run_dir = output_root / video_id
    started = time.perf_counter()

    asr_key = _read_user_environment("MIMO_API_KEY")
    llm_key = _read_user_environment("LLM_API_KEY")
    if not asr_key:
        asr_key = llm_key
    if not asr_key or not llm_key:
        raise RuntimeError(
            "MIMO_API_KEY/LLM_API_KEY is unavailable in the process or "
            "current-user environment."
        )

    detector_config = DetectorConfig.from_file(
        PROJECT_ROOT / "config" / "default.json"
    )
    transcription_config = replace(
        TranscriptionConfig.from_file(
            PROJECT_ROOT / "config" / "transcription.json"
        ),
        engine="mimo-cloud",
    )
    llm_config = LLMEvaluationConfig.from_file(
        PROJECT_ROOT / "config" / "llm_evaluation.json"
    )

    def elapsed() -> float:
        return round(time.perf_counter() - started, 3)

    def progress(stage: str, message: str, completed: int, total: int) -> None:
        print(
            json.dumps(
                {
                    "elapsed_sec": elapsed(),
                    "stage": stage,
                    "message": message,
                    "completed": completed,
                    "total": total,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    pipeline = TimedCloudPagePipeline(
        video_path=video,
        result_path=run_dir / "result.json",
        output_dir=run_dir,
        transcription_config=transcription_config,
        asr_api_key=asr_key,
        llm_config=llm_config,
        llm_api_key=llm_key,
        progress_callback=progress,
        clock_started_at=started,
    )
    try:
        detection = StreamingVideoPageDetector(detector_config).run(
            video,
            output_root=output_root,
            video_id=video_id,
            progress_callback=lambda message, value: print(
                json.dumps(
                    {
                        "elapsed_sec": elapsed(),
                        "stage": "detection",
                        "message": message,
                        "progress": value,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            ),
            page_ready_callback=pipeline.submit_page,
        )
        detection_completed_after_sec = elapsed()
        transcript, evaluation = pipeline.finish(detection)
    except BaseException:
        pipeline.abort()
        raise

    cloud_stats = transcript["transcription"]["cloud_statistics"]
    evaluations = evaluation["pages"] if evaluation else []
    summary = {
        "total_sec": elapsed(),
        "detection_completed_after_sec": detection_completed_after_sec,
        "first_page_submitted_after_sec": (
            pipeline.first_page_submitted_after_sec
        ),
        "first_asr_completed_after_sec": (
            pipeline.first_asr_completed_after_sec
        ),
        "first_llm_completed_after_sec": (
            pipeline.first_llm_completed_after_sec
        ),
        "page_count": len(detection["pages"]),
        "transcript_page_count": len(transcript["pages"]),
        "speech_character_count": sum(
            len(str(page.get("speech_text", "")))
            for page in transcript["pages"]
        ),
        "audio_chunk_request_count": cloud_stats[
            "audio_chunk_request_count"
        ],
        "uploaded_audio_bytes": cloud_stats["uploaded_audio_bytes"],
        "combined_cloud_concurrency_limit": cloud_stats[
            "combined_cloud_concurrency_limit"
        ],
        "evaluation_page_count": len(evaluations),
        "overall_score": (
            evaluation["summary"]["association_average_score"]
            if evaluation
            else None
        ),
    }
    (run_dir / "streaming_fullchain_performance.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
