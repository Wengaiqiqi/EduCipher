from __future__ import annotations

import json
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping

from .llm_evaluation import (
    PROMPT_VERSION,
    LLMEvaluationConfig,
    evaluate_page,
    render_evaluation_markdown,
    summarize_evaluations,
)
from .mimo_asr import MimoASRSettings, transcribe_pages_with_mimo
from .transcription import (
    TranscriptionConfig,
    render_page_transcripts_markdown,
)


PipelineProgressCallback = Callable[[str, str, int, int], None]
PageASRRunner = Callable[
    [Mapping[str, Any]],
    tuple[dict[str, Any], dict[str, Any]],
]
PageLLMRunner = Callable[[Mapping[str, Any]], dict[str, Any]]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class CloudPagePipeline:
    """Run page ASR and LLM evaluation as finalized pages become available."""

    def __init__(
        self,
        *,
        video_path: str | Path,
        result_path: str | Path,
        output_dir: str | Path,
        transcription_config: TranscriptionConfig,
        asr_api_key: str,
        llm_config: LLMEvaluationConfig | None = None,
        llm_api_key: str | None = None,
        progress_callback: PipelineProgressCallback | None = None,
        asr_runner: PageASRRunner | None = None,
        llm_runner: PageLLMRunner | None = None,
    ) -> None:
        transcription_config.validate()
        if transcription_config.engine != "mimo-cloud":
            raise ValueError("页面流水线仅适用于小米MiMo云端语音识别。")
        if not asr_api_key.strip() and asr_runner is None:
            raise ValueError("小米ASR API Key不能为空。")
        if llm_config is not None:
            llm_config.validate()
            if not (llm_api_key or "").strip() and llm_runner is None:
                raise ValueError("LLM API Key不能为空。")

        self.video_path = Path(video_path)
        self.result_path = Path(result_path)
        self.output_dir = Path(output_dir)
        self.transcription_config = transcription_config
        self.asr_api_key = asr_api_key
        self.llm_config = llm_config
        self.llm_api_key = llm_api_key or ""
        self.progress_callback = progress_callback
        self._asr_runner = asr_runner
        self._llm_runner = llm_runner
        self._lock = threading.Lock()
        self._cloud_concurrency_limit = max(
            transcription_config.mimo_max_concurrency,
            llm_config.max_concurrency if llm_config is not None else 1,
        )
        self._cloud_slots = threading.BoundedSemaphore(
            self._cloud_concurrency_limit
        )
        self._asr_executor = ThreadPoolExecutor(
            max_workers=transcription_config.mimo_max_concurrency,
            thread_name_prefix="page-pipeline-asr",
        )
        self._llm_executor = (
            ThreadPoolExecutor(
                max_workers=llm_config.max_concurrency,
                thread_name_prefix="page-pipeline-llm",
            )
            if llm_config is not None
            else None
        )
        self._asr_futures: list[Future[Any]] = []
        self._llm_futures: list[Future[Any]] = []
        self._page_transcripts: dict[int, dict[str, Any]] = {}
        self._asr_statistics: dict[int, dict[str, Any]] = {}
        self._evaluations: dict[int, dict[str, Any]] = {}
        self._asr_errors: dict[int, BaseException] = {}
        self._total_pages = 0
        self._asr_completed = 0
        self._llm_completed = 0
        self._first_asr_submitted_at: float | None = None
        self._last_asr_completed_at: float | None = None
        self._first_llm_submitted_at: float | None = None
        self._last_llm_completed_at: float | None = None
        self._closed = False

    def _report(
        self,
        stage: str,
        message: str,
        completed: int,
        total: int,
    ) -> None:
        if self.progress_callback is not None:
            self.progress_callback(stage, message, completed, total)

    def submit_page(
        self,
        page: Mapping[str, Any],
        completed: int,
        total: int,
    ) -> None:
        page_copy = dict(page)
        page_id = int(page_copy["page_id"])
        with self._lock:
            if self._closed:
                raise RuntimeError("页面流水线已经关闭。")
            self._total_pages = max(self._total_pages, total)
            if self._first_asr_submitted_at is None:
                self._first_asr_submitted_at = time.perf_counter()
            future = self._asr_executor.submit(
                self._transcribe_page,
                page_copy,
            )
            self._asr_futures.append(future)
        future.add_done_callback(
            lambda item, current_page_id=page_id: self._after_asr(
                current_page_id,
                item,
            )
        )
        self._report(
            "云端流水线",
            f"第{page_id}页已确认，立即提交小米语音识别",
            completed,
            total,
        )

    def _transcribe_page(
        self,
        page: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        with self._cloud_slots:
            if self._asr_runner is not None:
                return self._asr_runner(page)
            config = self.transcription_config
            settings = MimoASRSettings(
                base_url=config.mimo_base_url,
                model=config.mimo_model,
                language=config.mimo_language,
                max_concurrency=1,
                max_chunk_duration_sec=config.mimo_max_chunk_duration_sec,
                timeout_sec=config.mimo_timeout_sec,
                max_retries=config.mimo_max_retries,
                ffmpeg_path=config.ffmpeg_path,
            )
            pages, statistics = transcribe_pages_with_mimo(
                self.video_path,
                [page],
                settings=settings,
                api_key=self.asr_api_key,
            )
            return pages[0], statistics

    def _after_asr(
        self,
        page_id: int,
        future: Future[Any],
    ) -> None:
        try:
            page_transcript, statistics = future.result()
        except BaseException as exc:
            with self._lock:
                self._asr_errors[page_id] = exc
                self._asr_completed += 1
                completed = self._asr_completed
                total = self._total_pages
            self._report(
                "云端语音识别",
                f"第{page_id}页语音识别失败：{exc}",
                completed,
                total,
            )
            return

        with self._lock:
            self._page_transcripts[page_id] = dict(page_transcript)
            self._asr_statistics[page_id] = dict(statistics)
            self._asr_completed += 1
            self._last_asr_completed_at = time.perf_counter()
            completed = self._asr_completed
            total = self._total_pages
        self._report(
            "云端语音识别",
            f"第{page_id}页识别完成",
            completed,
            total,
        )

        if self._llm_executor is None:
            return
        with self._lock:
            if self._first_llm_submitted_at is None:
                self._first_llm_submitted_at = time.perf_counter()
            llm_future = self._llm_executor.submit(
                self._evaluate_page,
                dict(page_transcript),
            )
            self._llm_futures.append(llm_future)
        llm_future.add_done_callback(
            lambda item, current_page_id=page_id: self._after_llm(
                current_page_id,
                item,
            )
        )

    def _evaluate_page(
        self,
        page: Mapping[str, Any],
    ) -> dict[str, Any]:
        with self._cloud_slots:
            if self._llm_runner is not None:
                return self._llm_runner(page)
            assert self.llm_config is not None
            return evaluate_page(
                page,
                transcript_path=self.output_dir / "transcript.json",
                config=self.llm_config,
                output_dir=self.output_dir / "llm_evaluation",
                api_key=self.llm_api_key,
            )

    def _after_llm(
        self,
        page_id: int,
        future: Future[Any],
    ) -> None:
        try:
            evaluation = dict(future.result())
        except BaseException as exc:
            evaluation = {
                "page_id": page_id,
                "status": "failed",
                "speech_relevance": 0,
                "ppt_coverage": 0,
                "evidence_consistency": 0,
                "score": 0,
                "level": "请求失败",
                "reason": str(exc),
                "matched_evidence": [],
            }
        with self._lock:
            self._evaluations[page_id] = evaluation
            self._llm_completed += 1
            self._last_llm_completed_at = time.perf_counter()
            completed = self._llm_completed
            total = self._total_pages
        self._report(
            "LLM关联度评分",
            f"第{page_id}页评分完成",
            completed,
            total,
        )

    def finish(
        self,
        detection: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        with self._lock:
            self._closed = True
        self._asr_executor.shutdown(wait=True)
        if self._llm_executor is not None:
            self._llm_executor.shutdown(wait=True)

        pages = [
            dict(page)
            for page in detection.get("pages", [])
            if isinstance(page, Mapping)
        ]
        expected_ids = {int(page["page_id"]) for page in pages}
        if self._asr_errors:
            details = "; ".join(
                f"第{page_id}页：{error}"
                for page_id, error in sorted(self._asr_errors.items())
            )
            raise RuntimeError(f"页面流水线语音识别失败：{details}")
        if set(self._page_transcripts) != expected_ids:
            missing = sorted(expected_ids - set(self._page_transcripts))
            raise RuntimeError(f"页面流水线缺少语音结果：{missing}")

        transcript = self._write_transcript(detection, pages)
        evaluation = (
            self._write_evaluation(transcript)
            if self.llm_config is not None
            else None
        )
        return transcript, evaluation

    def abort(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._asr_executor.shutdown(wait=True, cancel_futures=True)
        if self._llm_executor is not None:
            self._llm_executor.shutdown(wait=True, cancel_futures=True)

    def _write_transcript(
        self,
        detection: Mapping[str, Any],
        pages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        ordered = [
            self._page_transcripts[int(page["page_id"])]
            for page in pages
        ]
        page_durations = [
            float(page.get("cloud_request_duration_sec", 0.0))
            for page in ordered
        ]
        statistics = list(self._asr_statistics.values())
        first = self._first_asr_submitted_at or time.perf_counter()
        last = self._last_asr_completed_at or first
        payload: dict[str, Any] = {
            "video_id": str(detection.get("video_id", self.video_path.stem)),
            "video_path": self.video_path.resolve().as_posix(),
            "video_duration_sec": detection.get("video_duration_sec"),
            "ppt_result_path": self.result_path.resolve().as_posix(),
            "processing_duration_sec": round(max(0.0, last - first), 3),
            "transcription": {
                "engine": "mimo-cloud",
                "model": self.transcription_config.mimo_model,
                "language": self.transcription_config.mimo_language,
                "language_probability": None,
                "speaker_diarization": False,
                "oral_filler_cleanup": False,
                "audio_files_retained": False,
                "utterance_count": sum(
                    len(page.get("utterances", [])) for page in ordered
                ),
                "cloud_statistics": {
                    "streaming_pipeline": True,
                    "page_request_count": len(ordered),
                    "audio_chunk_request_count": sum(
                        int(item.get("audio_chunk_request_count", 0))
                        for item in statistics
                    ),
                    "max_concurrency": (
                        self.transcription_config.mimo_max_concurrency
                    ),
                    "combined_cloud_concurrency_limit": (
                        self._cloud_concurrency_limit
                    ),
                    "max_chunk_duration_sec": (
                        self.transcription_config.mimo_max_chunk_duration_sec
                    ),
                    "uploaded_audio_bytes": sum(
                        int(item.get("uploaded_audio_bytes", 0))
                        for item in statistics
                    ),
                    "average_page_request_duration_sec": round(
                        sum(page_durations) / max(len(page_durations), 1),
                        3,
                    ),
                    "slowest_page_request_duration_sec": round(
                        max(page_durations, default=0.0),
                        3,
                    ),
                },
            },
            "pages": ordered,
            "config": asdict(self.transcription_config),
        }
        transcript_path = self.output_dir / "transcript.json"
        markdown_path = self.output_dir / "逐页语音文字.md"
        payload["artifacts"] = {
            "transcript_json": transcript_path.resolve().as_posix(),
            "page_transcript_markdown": markdown_path.resolve().as_posix(),
        }
        _write_json(transcript_path, payload)
        markdown_path.write_text(
            render_page_transcripts_markdown(
                payload["video_id"],
                ordered,
            ),
            encoding="utf-8",
        )
        return payload

    def _write_evaluation(
        self,
        transcript: Mapping[str, Any],
    ) -> dict[str, Any]:
        assert self.llm_config is not None
        expected_ids = {
            int(page["page_id"])
            for page in transcript.get("pages", [])
            if isinstance(page, Mapping)
        }
        if set(self._evaluations) != expected_ids:
            missing = sorted(expected_ids - set(self._evaluations))
            raise RuntimeError(
                f"页面流水线缺少LLM评分结果：{missing}"
            )
        ordered = [
            self._evaluations[page_id]
            for page_id in sorted(self._evaluations)
        ]
        summary = summarize_evaluations(ordered)
        first = self._first_llm_submitted_at or time.perf_counter()
        last = self._last_llm_completed_at or first
        destination = self.output_dir / "llm_evaluation"
        result_path = destination / "llm_evaluation.json"
        report_path = destination / "PPT讲话关联度报告.md"
        payload: dict[str, Any] = {
            "video_id": transcript["video_id"],
            "transcript_path": (
                self.output_dir / "transcript.json"
            ).resolve().as_posix(),
            "model": self.llm_config.model,
            "base_url": self.llm_config.base_url,
            "prompt_version": PROMPT_VERSION,
            "processing_duration_sec": round(max(0.0, last - first), 3),
            "summary": summary,
            "pages": ordered,
            "config": {
                key: value
                for key, value in asdict(self.llm_config).items()
                if key != "api_key_env"
            },
            "artifacts": {
                "result_json": result_path.resolve().as_posix(),
                "report_markdown": report_path.resolve().as_posix(),
                "page_results_dir": (
                    destination / "pages"
                ).resolve().as_posix(),
            },
        }
        _write_json(result_path, payload)
        report_path.write_text(
            render_evaluation_markdown(payload),
            encoding="utf-8",
        )
        return payload
