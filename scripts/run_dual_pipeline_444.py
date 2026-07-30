from __future__ import annotations

import json
import os
import sys
import threading
import time
import winreg
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from video_page_detector.analysis import detect_screen_crop_ratios  # noqa: E402
from video_page_detector.cloud_pipeline import CloudPagePipeline  # noqa: E402
from video_page_detector.config import DetectorConfig  # noqa: E402
from video_page_detector.dual_detector import (  # noqa: E402
    SceneHint,
    analyze_reuse,
    stream_scene_hints,
)
from video_page_detector.ffmpeg_io import FFmpegTools  # noqa: E402
from video_page_detector.llm_evaluation import (  # noqa: E402
    LLMEvaluationConfig,
)
from video_page_detector.mimo_asr import (  # noqa: E402
    MimoASRSettings,
    transcribe_pages_with_mimo,
)
from video_page_detector.pipeline import VideoPageDetector  # noqa: E402
from video_page_detector.transcription import (  # noqa: E402
    TranscriptionConfig,
)


def load_key(name: str, fallback: str | None = None) -> str:
    candidates = [name, *(tuple([fallback]) if fallback else ())]
    for candidate in candidates:
        value = os.environ.get(candidate, "").strip()
        if value:
            return value
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
        for candidate in candidates:
            try:
                value, _ = winreg.QueryValueEx(key, candidate)
            except FileNotFoundError:
                continue
            if isinstance(value, str) and value.strip():
                return value.strip()
    raise RuntimeError(f"没有找到用户级密钥：{name}")


def main() -> int:
    video = Path(r"E:\leeson\test_vedio\444.mp4")
    output_root = PROJECT_ROOT / "diagnostics" / "dual_pipeline_444"
    video_id = "444_dual_pipeline"
    run_dir = output_root / video_id
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
    asr_key = load_key("MIMO_API_KEY", "LLM_API_KEY")
    llm_key = load_key("LLM_API_KEY")
    started = time.perf_counter()
    timings: dict[str, float] = {}
    lock = threading.Lock()

    def elapsed() -> float:
        return round(time.perf_counter() - started, 3)

    def emit(stage: str, message: str, **extra: Any) -> None:
        print(
            json.dumps(
                {
                    "elapsed_sec": elapsed(),
                    "stage": stage,
                    "message": message,
                    **extra,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    tools = FFmpegTools(
        ffmpeg_path=detector_config.ffmpeg_path,
        ffprobe_path=detector_config.ffprobe_path,
        analysis_width=detector_config.temporal_analysis_width,
        analysis_height=detector_config.temporal_analysis_height,
    )
    metadata = tools.probe(video)
    fallback_crop = (
        detector_config.screen_crop_left_ratio,
        detector_config.screen_crop_top_ratio,
        detector_config.screen_crop_right_ratio,
        detector_config.screen_crop_bottom_ratio,
    )
    calibration_frames = []
    for fraction in (0.1, 0.3, 0.5, 0.7, 0.9):
        frames = tools.sample_window(
            video,
            start_sec=max(0.0, metadata.duration_sec * fraction - 0.5),
            duration_sec=1.0,
            fps=1.0,
        )
        if frames:
            calibration_frames.append(frames[0])
    crop_ratios = detect_screen_crop_ratios(
        calibration_frames,
        fallback=fallback_crop,
    )
    timings["shared_crop_ready_sec"] = elapsed()
    emit("双算法", "共享投影区域校准完成")

    mimo_settings = MimoASRSettings(
        base_url=transcription_config.mimo_base_url,
        model=transcription_config.mimo_model,
        language=transcription_config.mimo_language,
        max_concurrency=1,
        max_chunk_duration_sec=(
            transcription_config.mimo_max_chunk_duration_sec
        ),
        timeout_sec=transcription_config.mimo_timeout_sec,
        max_retries=transcription_config.mimo_max_retries,
        ffmpeg_path=transcription_config.ffmpeg_path,
    )
    speculative_executor = ThreadPoolExecutor(
        max_workers=transcription_config.mimo_max_concurrency,
        thread_name_prefix="dual-speculative-asr",
    )
    speculative_futures: dict[
        Future[tuple[dict[str, Any], dict[str, Any]]],
        tuple[float, float],
    ] = {}
    speculative_results: dict[
        tuple[float, float],
        tuple[dict[str, Any], dict[str, Any]],
    ] = {}
    speculative_errors: dict[tuple[float, float], str] = {}
    accepted_hints: list[SceneHint] = []
    raw_hints: list[SceneHint] = []
    previous_boundary = 0.0

    def interval_key(start_sec: float, end_sec: float) -> tuple[float, float]:
        return round(start_sec, 3), round(end_sec, 3)

    def transcribe_interval(
        page: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        pages, statistics = transcribe_pages_with_mimo(
            video,
            [page],
            settings=mimo_settings,
            api_key=asr_key,
        )
        return pages[0], statistics

    def after_speculative(
        future: Future[tuple[dict[str, Any], dict[str, Any]]],
    ) -> None:
        key = speculative_futures[future]
        try:
            result = future.result()
        except BaseException as exc:
            with lock:
                speculative_errors[key] = str(exc)
            emit(
                "提前ASR",
                f"候选区间{key[0]:.3f}～{key[1]:.3f}识别失败",
            )
            return
        with lock:
            speculative_results[key] = result
            timings.setdefault("first_speculative_asr_completed_sec", elapsed())
            timings["last_speculative_asr_completed_sec"] = elapsed()
        emit(
            "提前ASR",
            f"候选区间{key[0]:.3f}～{key[1]:.3f}识别完成",
        )

    def submit_interval(start_sec: float, end_sec: float) -> None:
        if end_sec <= start_sec:
            return
        key = interval_key(start_sec, end_sec)
        page = {
            "page_id": len(speculative_futures) + 1001,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "screenshot_path": None,
        }
        with lock:
            timings.setdefault("first_speculative_asr_submitted_sec", elapsed())
            future = speculative_executor.submit(transcribe_interval, page)
            speculative_futures[future] = key
        future.add_done_callback(after_speculative)
        emit(
            "提前ASR",
            f"场景区间{start_sec:.3f}～{end_sec:.3f}立即提交",
        )

    def on_scene_hint(hint: SceneHint) -> None:
        nonlocal previous_boundary
        raw_hints.append(hint)
        if hint.timestamp_sec - previous_boundary < 5.0:
            emit(
                "场景提示",
                f"{hint.timestamp_sec:.3f}秒候选过短，暂不采用",
            )
            return
        accepted_hints.append(hint)
        submit_interval(previous_boundary, hint.timestamp_sec)
        previous_boundary = hint.timestamp_sec
        with lock:
            timings.setdefault("first_scene_hint_sec", elapsed())
        emit(
            "场景提示",
            f"确认快速候选边界{hint.timestamp_sec:.3f}秒",
        )

    def scan_scenes() -> list[SceneHint]:
        nonlocal previous_boundary
        hints = stream_scene_hints(
            video,
            threshold=0.05,
            crop_ratios=crop_ratios,
            ffmpeg_path=detector_config.ffmpeg_path,
            callback=on_scene_hint,
        )
        submit_interval(previous_boundary, metadata.duration_sec)
        timings["scene_scan_completed_sec"] = elapsed()
        emit("场景提示", "流式场景扫描完成")
        return hints

    scene_executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="dual-scene-scan",
    )
    scene_future = scene_executor.submit(scan_scenes)
    detection = VideoPageDetector(detector_config).run(
        video,
        output_root=output_root,
        video_id=video_id,
        progress_callback=lambda message, progress: emit(
            "时序算法",
            message,
            progress=progress,
        ),
    )
    timings["temporal_detection_completed_sec"] = elapsed()
    scene_future.result()
    scene_executor.shutdown(wait=True)
    speculative_executor.shutdown(wait=True)
    timings["reconciliation_started_sec"] = elapsed()
    emit("双算法", "时序结果与提前ASR均已就绪，开始校正")

    reuse_report = analyze_reuse(
        detection["pages"],
        accepted_hints,
        video_duration_sec=float(detection["video_duration_sec"]),
        tolerance_sec=2.0,
    )
    decision_by_page = {
        int(item["page_id"]): item
        for item in reuse_report["page_decisions"]
    }
    exact_reprocess_stats: list[dict[str, Any]] = []
    reused_page_ids: list[int] = []
    reprocessed_page_ids: list[int] = []

    def align_cached_transcript(
        transcript: Mapping[str, Any],
        final_page: Mapping[str, Any],
    ) -> dict[str, Any]:
        item = dict(transcript)
        item.update(
            {
                "page_id": int(final_page["page_id"]),
                "start_sec": round(float(final_page["start_sec"]), 3),
                "end_sec": round(float(final_page["end_sec"]), 3),
                "screenshot_path": final_page.get("screenshot_path"),
                "speculative_reused": True,
                "speculative_start_sec": transcript.get("start_sec"),
                "speculative_end_sec": transcript.get("end_sec"),
            }
        )
        utterances = [
            dict(utterance)
            for utterance in item.get("utterances", [])
        ]
        if utterances:
            utterances[0]["start_sec"] = item["start_sec"]
            utterances[-1]["end_sec"] = item["end_sec"]
        item["utterances"] = utterances
        return item

    def final_asr_runner(
        page: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        page_id = int(page["page_id"])
        decision = decision_by_page[page_id]
        if decision["reusable"]:
            key = interval_key(
                float(decision["provisional_start_sec"]),
                float(decision["provisional_end_sec"]),
            )
            cached = speculative_results.get(key)
            if cached is not None:
                reused_page_ids.append(page_id)
                transcript, statistics = cached
                return (
                    align_cached_transcript(transcript, page),
                    statistics,
                )
        reprocessed_page_ids.append(page_id)
        transcript, statistics = transcribe_interval(page)
        transcript = dict(transcript)
        transcript["speculative_reused"] = False
        exact_reprocess_stats.append(statistics)
        return transcript, statistics

    def final_progress(
        stage: str,
        message: str,
        completed: int,
        total: int,
    ) -> None:
        if stage == "LLM关联度评分":
            timings.setdefault("first_final_llm_completed_sec", elapsed())
        emit(
            stage,
            message,
            completed=completed,
            total=total,
        )

    final_pipeline = CloudPagePipeline(
        video_path=video,
        result_path=run_dir / "result.json",
        output_dir=run_dir,
        transcription_config=transcription_config,
        asr_api_key="",
        llm_config=llm_config,
        llm_api_key=llm_key,
        progress_callback=final_progress,
        asr_runner=final_asr_runner,
    )
    try:
        total_pages = len(detection["pages"])
        for index, page in enumerate(detection["pages"], start=1):
            final_pipeline.submit_page(page, index, total_pages)
        transcript, evaluation = final_pipeline.finish(detection)
    except BaseException:
        final_pipeline.abort()
        raise

    timings["all_completed_sec"] = elapsed()
    speculative_statistics = [
        statistics
        for _, statistics in speculative_results.values()
    ]
    performance = {
        **timings,
        "page_count": len(detection["pages"]),
        "raw_scene_hint_count": len(raw_hints),
        "accepted_scene_hint_count": len(accepted_hints),
        "matched_boundary_count": reuse_report["matched_boundary_count"],
        "boundary_match_rate": reuse_report["boundary_match_rate"],
        "planned_reusable_page_count": reuse_report["reusable_page_count"],
        "actual_reused_page_count": len(reused_page_ids),
        "reused_page_ids": sorted(reused_page_ids),
        "reprocessed_page_count": len(reprocessed_page_ids),
        "reprocessed_page_ids": sorted(reprocessed_page_ids),
        "speculative_interval_count": len(speculative_futures),
        "speculative_failed_interval_count": len(speculative_errors),
        "speculative_audio_chunk_request_count": sum(
            int(item.get("audio_chunk_request_count", 0))
            for item in speculative_statistics
        ),
        "speculative_uploaded_audio_bytes": sum(
            int(item.get("uploaded_audio_bytes", 0))
            for item in speculative_statistics
        ),
        "exact_reprocess_audio_chunk_request_count": sum(
            int(item.get("audio_chunk_request_count", 0))
            for item in exact_reprocess_stats
        ),
        "exact_reprocess_uploaded_audio_bytes": sum(
            int(item.get("uploaded_audio_bytes", 0))
            for item in exact_reprocess_stats
        ),
        "transcript_characters": sum(
            len(page["speech_text"]) for page in transcript["pages"]
        ),
        "score": evaluation["summary"]["strict_overall_score"],
        "baseline_v1_3_total_sec": 366.612,
        "saved_vs_baseline_sec": round(
            366.612 - timings["all_completed_sec"],
            3,
        ),
    }
    reuse_report["actual_reused_page_ids"] = sorted(reused_page_ids)
    reuse_report["actual_reprocessed_page_ids"] = sorted(
        reprocessed_page_ids
    )
    (run_dir / "dual_detector_report.json").write_text(
        json.dumps(reuse_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "dual_pipeline_performance.json").write_text(
        json.dumps(performance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(performance, ensure_ascii=False, indent=2),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
