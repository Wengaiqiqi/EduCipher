from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from .cloud_pipeline import CloudPagePipeline
from .config import DetectorConfig
from .llm_evaluation import LLMEvaluationConfig, evaluate_transcript
from .mimo_asr import resolve_mimo_api_key
from .output_paths import resolve_run_directory, validate_video_id
from .pipeline import VideoPageDetector
from .transcription import TranscriptionConfig, transcribe_video_pages


_output_lock = threading.Lock()
_task_lock = threading.Lock()
_cancel_event = threading.Event()
_active_thread: threading.Thread | None = None


def emit(event_type: str, **payload: Any) -> None:
    message = {"type": event_type, **payload}
    with _output_lock:
        line = json.dumps(message, ensure_ascii=False) + "\n"
        try:
            sys.stdout.write(line)
            sys.stdout.flush()
        except UnicodeEncodeError:
            if hasattr(sys.stdout, "buffer"):
                sys.stdout.buffer.write(line.encode("utf-8", "replace"))
                sys.stdout.buffer.flush()
            else:
                sys.stdout.write(
                    line.encode("utf-8", "replace").decode("ascii", "replace")
                )
                sys.stdout.flush()


def project_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def resource_path(relative: str | Path) -> Path:
    bundle = getattr(sys, "_MEIPASS", None)
    candidates = [
        Path(bundle) / relative if bundle else None,
        project_root() / relative,
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    return project_root() / relative


def bundled_tool(name: str) -> str | None:
    suffix = ".exe" if os.name == "nt" else ""
    candidate = resource_path(Path("tools") / f"{name}{suffix}")
    return str(candidate) if candidate.is_file() else None


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def run_metadata_path(run_dir: Path) -> Path:
    return run_dir / "run_metadata.json"


def write_run_metadata(
    run_dir: Path,
    started_at: float,
    *,
    mode: str,
    include_llm: bool,
) -> None:
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        run_metadata_path(run_dir).write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "started_at": started_at,
                    "status": "running",
                    "mode": mode,
                    "include_llm": include_llm,
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    except OSError:
        pass


def load_run_elapsed(run_dir: Path) -> float | None:
    maximum_reasonable_elapsed = 30 * 24 * 60 * 60
    try:
        data = json.loads(run_metadata_path(run_dir).read_text(encoding="utf-8"))
        # 优先使用任务完成时保存的精确耗时
        if "elapsed_sec" in data:
            elapsed = float(data["elapsed_sec"])
            if 0 <= elapsed <= maximum_reasonable_elapsed:
                return round(elapsed, 3)
        # 旧版：从 Unix started_at 和最后一个结果文件的修改时间推算。
        started_at = float(data.get("started_at") or 0)
        result_path = run_dir / "result.json"
        if not result_path.exists():
            return None
        artifact_paths = [
            result_path,
            run_dir / "transcript.json",
            run_dir / "llm_evaluation" / "llm_evaluation.json",
        ]
        completed_at = max(
            path.stat().st_mtime
            for path in artifact_paths
            if path.is_file()
        )
        if 1e9 <= started_at <= completed_at:
            elapsed = completed_at - started_at
            if 0 < elapsed <= maximum_reasonable_elapsed:
                return round(elapsed, 3)

        # 更早的版本误把 perf_counter 写入 started_at。此时使用
        # result.json 的写入时间减去页面检测耗时，重建任务开始时间。
        detection = read_json(result_path) or {}
        try:
            detection_elapsed = float(
                detection.get("processing_duration_sec") or 0
            )
        except (TypeError, ValueError):
            detection_elapsed = 0
        if 0 < detection_elapsed <= maximum_reasonable_elapsed:
            inferred_started_at = result_path.stat().st_mtime - detection_elapsed
            elapsed = completed_at - inferred_started_at
            if 0 < elapsed <= maximum_reasonable_elapsed:
                return round(elapsed, 3)
        return None
    except Exception:
        return None



def page_speech_text(page: Mapping[str, Any]) -> str:
    explicit = str(page.get("speech_text") or "").strip()
    if explicit:
        return explicit
    utterances = page.get("utterances", [])
    if not isinstance(utterances, list):
        return ""
    return "\n".join(
        str(item.get("text") or "").strip()
        for item in utterances
        if isinstance(item, Mapping) and str(item.get("text") or "").strip()
    )


def resolve_screenshot_path(page: Mapping[str, Any], run_dir: Path) -> str:
    """Resolve screenshot_path to an absolute Windows path string suitable for convertFileSrc."""
    raw = str(page.get("screenshot_path") or "").strip()
    if not raw:
        return ""
    candidate = Path(raw)
    if candidate.is_absolute():
        return str(candidate.resolve())
    return str((run_dir / candidate.name).resolve())


def load_task(run_dir: Path) -> dict[str, Any] | None:
    detection = read_json(run_dir / "result.json")
    if not detection or not isinstance(detection.get("pages"), list):
        return None
    transcript = read_json(run_dir / "transcript.json")
    evaluation = read_json(
        run_dir / "llm_evaluation" / "llm_evaluation.json"
    )
    metadata = read_json(run_metadata_path(run_dir)) or {}
    evaluation_pages = evaluation.get("pages", []) if evaluation else []
    evidence_enabled = bool(
        evaluation
        and (
            evaluation.get("config", {}).get("include_evidence", False)
            or any(
                isinstance(page, Mapping)
                and "matched_evidence" in page
                for page in evaluation_pages
            )
        )
    )
    page_map: dict[int, dict[str, Any]] = {}
    for page in detection["pages"]:
        if not isinstance(page, Mapping) or "page_id" not in page:
            continue
        page_id = int(page["page_id"])
        page_map[page_id] = {**dict(page), "status": "detected"}
        row = page_map[page_id]
        row["screenshot_path"] = resolve_screenshot_path(page, run_dir)
    if transcript:
        for page in transcript.get("pages", []):
            if not isinstance(page, Mapping) or "page_id" not in page:
                continue
            page_id = int(page["page_id"])
            row = page_map.setdefault(page_id, {"page_id": page_id})
            # preserve resolved screenshot_path from detection
            existing_screenshot = row.get("screenshot_path", "")
            existing_speech = row.get("speech_text", "")
            row.update(dict(page))
            if existing_screenshot:
                row["screenshot_path"] = existing_screenshot
            row["speech_text"] = existing_speech or page_speech_text(page)
            row["status"] = "detected"
    if evaluation:
        for page in evaluation.get("pages", []):
            if not isinstance(page, Mapping) or "page_id" not in page:
                continue
            page_id = int(page["page_id"])
            row = page_map.setdefault(page_id, {"page_id": page_id})
            # evaluation dict lacks screenshot_path/speech_text, so preserve existing
            for key, value in page.items():
                if key in {"screenshot_path", "speech_text"}:
                    continue
                row[key] = value
            # rename matched_evidence to evidence for frontend;
            # 如果 matched_evidence 不存在（任务运行时 include_evidence=False），
            # 显式设为 None 以便前端区分"未生成"和"生成为空"
            if "matched_evidence" in row:
                row["evidence"] = row.pop("matched_evidence")
            elif "evidence" not in row:
                row["evidence"] = None
            row["status"] = (
                "failed" if page.get("status") == "failed" else "completed"
            )
    try:
        updated_at = (run_dir / "result.json").stat().st_mtime * 1000
    except OSError:
        updated_at = 0
    elapsed = load_run_elapsed(run_dir)
    evaluation_summary = evaluation.get("summary", {}) if evaluation else {}
    failed_pages = int(evaluation_summary.get("failed_pages") or 0)
    evaluation_complete = evaluation_summary.get("complete")
    metadata_status = str(metadata.get("status") or "")
    completed_with_errors = bool(
        evaluation
        and (failed_pages > 0 or evaluation_complete is False)
    ) or metadata_status == "completed_with_errors"
    if completed_with_errors:
        task_status = "completed_with_errors"
    elif metadata_status == "completed" or evaluation:
        task_status = "completed"
    elif metadata_status in {"failed", "cancelled"}:
        task_status = "failed"
    else:
        task_status = "idle"
    finished = task_status in {"completed", "completed_with_errors"}
    mode = str(
        metadata.get("mode")
        or ("full" if transcript or evaluation else "detect")
    )
    include_llm = bool(metadata.get("include_llm", bool(evaluation)))
    completed_stages = ["ppt"]
    if transcript:
        completed_stages.append("voice")
    if evaluation:
        completed_stages.append("llm")
    if finished:
        completed_stages.append("report")
    return {

        "id": str(run_dir.resolve()),
        "video_id": str(detection.get("video_id") or run_dir.name),
        "video_path": str(detection.get("video_path") or ""),
        "run_dir": str(run_dir.resolve()),
        "updated_at": updated_at,
        "status": task_status,
        "progress": 100 if finished else (78 if transcript else 36),
        "stage": (
            "处理完成，但部分页面存在错误"
            if completed_with_errors
            else (
                "处理完成"
                if task_status == "completed"
                else (
                    "处理失败"
                    if task_status == "failed"
                    else (
                        "等待关联度评分"
                        if transcript
                        else "PPT页面识别完成"
                    )
                )
            )
        ),
        "elapsed_sec": elapsed,
        "model": str(evaluation.get("model") or "") if evaluation else "",
        "include_evidence": evidence_enabled,
        "include_llm": include_llm,
        "mode": mode,
        "completed_stages": completed_stages,
        "summary": (
            evaluation.get("summary", {})
            if evaluation
            else {"total_pages": len(page_map)}
        ),
        "pages": [page_map[key] for key in sorted(page_map)],
    }


def list_tasks(output_root: str | None = None) -> list[dict[str, Any]]:
    roots = [
        Path(output_root) if output_root else None,
        project_root() / "output",
        Path.home() / "Documents" / "课堂PPT处理结果",
    ]
    tasks: dict[str, dict[str, Any]] = {}
    for root in roots:
        if root is None or not root.is_dir():
            continue
        try:
            children = list(root.iterdir())
        except OSError:
            continue
        for child in children:
            if not child.is_dir():
                continue
            task = load_task(child)
            if task:
                tasks[str(child.resolve()).lower()] = task
    return sorted(
        tasks.values(),
        key=lambda item: float(item.get("updated_at") or 0),
        reverse=True,
    )[:30]


def require_not_cancelled() -> None:
    if _cancel_event.is_set():
        raise RuntimeError("任务已由用户取消")


def default_detector_config(settings: Mapping[str, Any]) -> DetectorConfig:
    config_path = resource_path("config/default.json")
    config = (
        DetectorConfig.from_file(config_path)
        if config_path.is_file()
        else DetectorConfig()
    )
    preset = str(settings.get("detector_preset") or "precise")
    if preset == "fast":
        config = replace(
            config,
            temporal_sample_interval_sec=4.0,
            temporal_confirmation_sec=12.0,
            temporal_analysis_width=256,
            temporal_analysis_height=144,
            temporal_refinement_fps=2.0,
            jpeg_quality=85,
        )
    config = replace(
        config,
        ffmpeg_path=bundled_tool("ffmpeg") or config.ffmpeg_path,
        ffprobe_path=bundled_tool("ffprobe") or config.ffprobe_path,
    )
    config.validate()
    return config


def transcription_config(settings: Mapping[str, Any]) -> TranscriptionConfig:
    config_path = resource_path("config/transcription.json")
    base = (
        TranscriptionConfig.from_file(config_path)
        if config_path.is_file()
        else TranscriptionConfig()
    )
    config = replace(
        base,
        engine=str(settings.get("asr_engine") or base.engine),
        model=str(settings.get("asr_model") or base.model),
        mimo_base_url=str(
            settings.get("mimo_base_url") or base.mimo_base_url
        ),
        mimo_model=str(settings.get("mimo_model") or base.mimo_model),
        mimo_max_concurrency=min(
            10,
            int(settings.get("asr_concurrency") or base.mimo_max_concurrency),
        ),
        model_download_root=str(resource_path("models/faster-whisper")),
        ffmpeg_path=bundled_tool("ffmpeg") or base.ffmpeg_path,
    )
    config.validate()
    return config


def llm_config(settings: Mapping[str, Any]) -> LLMEvaluationConfig:
    config_path = resource_path("config/llm_evaluation.json")
    base = (
        LLMEvaluationConfig.from_file(config_path)
        if config_path.is_file()
        else LLMEvaluationConfig()
    )
    ev = bool(settings.get("include_evidence", False))
    config = replace(
        base,
        base_url=str(settings.get("llm_base_url") or base.base_url),
        model=str(settings.get("llm_model") or base.model),
        max_concurrency=min(
            10,
            int(settings.get("llm_concurrency") or base.max_concurrency),
        ),
        include_evidence=ev,
    )
    config.validate()
    return config


def emit_page(
    page: Mapping[str, Any],
    status: str,
    task_id: str = "",
    **extra: Any,
) -> None:
    payload = dict(page)
    # 过滤掉值为 None 的 extra 字段，避免前端收到 null 覆盖已有数据
    payload.update((k, v) for k, v in extra.items() if v is not None)
    payload["status"] = status
    if "speech_text" not in payload:
        payload["speech_text"] = page_speech_text(payload)
    emit("page.updated", task_id=task_id, page=payload)


def run_task(payload: Mapping[str, Any]) -> None:
    started_at = time.perf_counter()
    cloud_pipeline: CloudPagePipeline | None = None
    task_id = ""
    run_dir: Path | None = None
    try:
        video = Path(str(payload.get("video_path") or ""))
        if not video.is_file():
            raise ValueError("请选择有效的课堂视频。")
        output_root = Path(str(payload.get("output_root") or ""))
        if not str(output_root):
            raise ValueError("请选择结果保存目录。")
        output_root.mkdir(parents=True, exist_ok=True)
        video_id = validate_video_id(str(payload.get("video_id") or video.stem))
        mode = str(payload.get("mode") or "full")
        settings = payload.get("settings", {})
        if not isinstance(settings, Mapping):
            raise ValueError("桌面端设置格式不正确。")
        run_dir = resolve_run_directory(output_root, video_id)
        result_path = run_dir / "result.json"
        transcript_path = run_dir / "transcript.json"
        evaluation_dir = run_dir / "llm_evaluation"
        task_id = str(run_dir.resolve())
        include_llm = bool(settings.get("include_llm", True)) and mode == "full"

        emit(
            "task.started",
            task_id=task_id,
            video_id=video_id,
            video_path=str(video.resolve()),
            run_dir=str(run_dir.resolve()),
            started_at=time.time(),
            algorithm_version="1.4.1",
            streaming_page_confirmation=True,
            mode=mode,
            include_llm=include_llm,
            include_evidence=bool(settings.get("include_evidence", False)),
        )
        detector = VideoPageDetector(default_detector_config(settings))
        write_run_metadata(
            run_dir,
            time.time(),
            mode=mode,
            include_llm=include_llm,
        )
        transcribe_config = transcription_config(settings) if mode == "full" else None
        llm_settings = llm_config(settings) if include_llm else None
        asr_key = ""
        llm_key = str(payload.get("llm_api_key") or "").strip()

        if transcribe_config and transcribe_config.engine == "mimo-cloud":
            if not bool(payload.get("asr_upload_consent")):
                raise ValueError("请确认允许把临时音频发送给小米 MiMo。")
            asr_key = resolve_mimo_api_key(
                transcribe_config.mimo_api_key_env,
                str(payload.get("asr_api_key") or ""),
            )
        if include_llm:
            if not bool(payload.get("llm_upload_consent")):
                raise ValueError("请确认允许发送 PPT 截图和课堂转写。")
            llm_key = llm_key or os.environ.get("LLM_API_KEY", "").strip()
            if not llm_key:
                raise ValueError("没有找到 LLM API Key。")

        streaming_cloud = (
            mode == "full"
            and transcribe_config is not None
            and transcribe_config.engine == "mimo-cloud"
        )

        if streaming_cloud:
            def pipeline_progress(
                stage: str,
                message: str,
                completed: int,
                total: int,
            ) -> None:
                ratio = completed / max(total, 1)
                progress = (
                    58 + ratio * 40
                    if stage == "LLM关联度评分"
                    else 32 + ratio * (45 if include_llm else 66)
                )
                emit(
                    "task.progress",
                    task_id=task_id,
                    stage=stage,
                    message=message,
                    progress=progress,
                    stage_progress=round(ratio * 100),
                    completed=completed,
                    total=total,
                )
                match = re.search(r"第(\d+)页", message)
                if not match or cloud_pipeline is None:
                    return
                page_id = int(match.group(1))
                if stage in {"云端语音识别", "语音转写"}:
                    page = cloud_pipeline._page_transcripts.get(page_id)
                    if page:
                        emit_page(page, "scoring" if include_llm else "completed", task_id=task_id)
                elif stage == "LLM关联度评分":
                    evaluation = cloud_pipeline._evaluations.get(page_id)
                    transcript = cloud_pipeline._page_transcripts.get(
                        page_id, {"page_id": page_id}
                    )
                    if evaluation:
                        kwargs = dict(
                            score=evaluation.get("score"),
                            level=evaluation.get("level"),
                            reason=evaluation.get("reason"),
                        )
                        matched = evaluation.get("matched_evidence")
                        if matched is not None:
                            kwargs["evidence"] = matched
                        emit_page(transcript, "completed", task_id=task_id, **kwargs)

            cloud_pipeline = CloudPagePipeline(
                video_path=video,
                result_path=result_path,
                output_dir=run_dir,
                transcription_config=transcribe_config,
                asr_api_key=asr_key,
                llm_config=llm_settings,
                llm_api_key=llm_key,
                progress_callback=pipeline_progress,
                cloud_activity_callback=lambda active, limit: emit(
                    "cloud.activity",
                    task_id=task_id,
                    active_cloud_requests=active,
                    cloud_limit=limit,
                ),
                cancel_event=_cancel_event,
            )

        def detection_progress(message: str, progress: float | None) -> None:
            require_not_cancelled()
            emit(
                "task.progress",
                task_id=task_id,
                stage="PPT页面识别",
                message=message,
                progress=float(progress or 0)
                * (58 if mode == "full" else 100),
                stage_progress=round(float(progress or 0) * 100),
            )

        def page_ready(page: dict[str, Any], completed: int, total: int) -> None:
            require_not_cancelled()
            emit_page(
                page,
                "transcribing" if cloud_pipeline else "detected",
                task_id=task_id,
            )
            emit(
                "task.progress",
                task_id=task_id,
                stage="PPT页面识别",
                message=f"第{page['page_id']}页高清截图已确认",
                completed=completed,
                total=total,
            )
            if cloud_pipeline:
                cloud_pipeline.submit_page(page, completed, total)

        detection = detector.run(
            video,
            output_root=output_root,
            video_id=video_id,
            progress_callback=detection_progress,
            page_ready_callback=page_ready,
        )
        require_not_cancelled()
        emit(
            "task.progress",
            task_id=task_id,
            stage="PPT页面识别",
            message=f"PPT页面识别完成，共确认{len(detection.get('pages', []))}页",
            progress=58 if mode == "full" else 100,
            stage_progress=100,
            completed_stage="ppt",
        )
        transcript: dict[str, Any] | None = None
        evaluation: dict[str, Any] | None = None
        if cloud_pipeline:
            emit(
                "task.progress",
                task_id=task_id,
                stage="云端流水线",
                message="页面识别完成，正在等待剩余转写与评分",
                progress=60,
            )
            transcript, evaluation = cloud_pipeline.finish(detection)
            cloud_pipeline = None
        elif mode == "full" and transcribe_config:
            def transcription_progress(
                message: str,
                progress: float | None,
            ) -> None:
                require_not_cancelled()
                emit(
                    "task.progress",
                    task_id=task_id,
                    stage="语音转写",
                    message=message,
                    progress=30 + float(progress or 0) * 45,
                    stage_progress=round(float(progress or 0) * 100),
                )

            transcript = transcribe_video_pages(
                video,
                result_path,
                config=transcribe_config,
                output_dir=run_dir,
                api_key=str(payload.get("asr_api_key") or "") or None,
                progress_callback=transcription_progress,
            )
            for page in transcript.get("pages", []):
                if isinstance(page, Mapping):
                    emit_page(
                        page,
                        "scoring" if include_llm else "completed",
                        task_id=task_id,
                    )
            require_not_cancelled()
            if include_llm and llm_settings:
                def evaluation_progress(
                    message: str,
                    completed: int,
                    total: int,
                ) -> None:
                    require_not_cancelled()
                    emit(
                        "task.progress",
                        task_id=task_id,
                        stage="LLM关联度评分",
                        message=f"{completed}/{total}：{message}",
                        progress=75 + completed / max(total, 1) * 25,
                        stage_progress=round(completed / max(total, 1) * 100),
                    )

                evaluation = evaluate_transcript(
                    transcript_path,
                    config=llm_settings,
                    output_dir=evaluation_dir,
                    api_key=llm_key,
                    progress_callback=evaluation_progress,
                    activity_callback=lambda active, limit: emit(
                        "cloud.activity",
                        task_id=task_id,
                        active_cloud_requests=active,
                        cloud_limit=limit,
                    ),
                )
                evaluation_by_id = {
                    int(item["page_id"]): item
                    for item in evaluation.get("pages", [])
                    if isinstance(item, Mapping) and "page_id" in item
                }
                for page in transcript.get("pages", []):
                    if not isinstance(page, Mapping):
                        continue
                    item = evaluation_by_id.get(int(page["page_id"]), {})
                    kwargs = dict(
                        score=item.get("score"),
                        level=item.get("level"),
                        reason=item.get("reason"),
                    )
                    matched = item.get("matched_evidence")
                    if matched is not None:
                        kwargs["evidence"] = matched
                    emit_page(page, "completed", task_id=task_id, **kwargs)

        elapsed_sec = round(time.perf_counter() - started_at, 3)
        # 先把精确耗时保存到 run_metadata.json，再 load_task，确保耗时能被读到
        try:
            meta = read_json(run_metadata_path(run_dir)) or {}
            meta["elapsed_sec"] = elapsed_sec
            meta["completed_at"] = time.time()
            evaluation_summary = evaluation.get("summary", {}) if evaluation else {}
            meta["status"] = (
                "completed_with_errors"
                if evaluation
                and (
                    int(evaluation_summary.get("failed_pages") or 0) > 0
                    or evaluation_summary.get("complete") is False
                )
                else "completed"
            )
            run_metadata_path(run_dir).write_text(
                json.dumps(meta, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except OSError:
            pass
        task = load_task(run_dir)
        emit(
            "task.completed",
            task_id=task_id,
            elapsed_sec=elapsed_sec,
            result=task,
        )
    except Exception as exc:
        if cloud_pipeline is not None:
            cloud_pipeline.abort()
        if run_dir is not None:
            try:
                meta = read_json(run_metadata_path(run_dir)) or {}
                meta["elapsed_sec"] = round(time.perf_counter() - started_at, 3)
                meta["completed_at"] = time.time()
                meta["status"] = (
                    "cancelled" if _cancel_event.is_set() else "failed"
                )
                run_metadata_path(run_dir).write_text(
                    json.dumps(meta, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
            except OSError:
                pass
        emit(
            "task.failed",
            task_id=task_id,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
    finally:
        _cancel_event.clear()


def start_task(payload: Mapping[str, Any]) -> None:
    global _active_thread
    with _task_lock:
        if _active_thread is not None and _active_thread.is_alive():
            emit("task.failed", error="已有任务正在处理中。")
            return
        _cancel_event.clear()
        _active_thread = threading.Thread(
            target=run_task,
            args=(dict(payload),),
            daemon=True,
            name="desktop-v2-task",
        )
        _active_thread.start()


def handle_command(command: Mapping[str, Any]) -> None:
    action = str(command.get("action") or "")
    if action == "ping":
        emit(
            "worker.ready",
            project_root=str(project_root()),
            algorithm_version="1.4.1",
            ffmpeg_path=bundled_tool("ffmpeg") or "PATH",
            ffprobe_path=bundled_tool("ffprobe") or "PATH",
        )
    elif action == "start":
        payload = command.get("payload", {})
        if not isinstance(payload, Mapping):
            emit("task.failed", error="任务参数格式不正确。")
            return
        start_task(payload)
    elif action == "cancel":
        _cancel_event.set()
        emit("task.cancelling", message="将在当前处理步骤完成后取消任务。")
    elif action == "list_tasks":
        emit(
            "tasks.list",
            tasks=list_tasks(str(command.get("output_root") or "") or None),
        )
    else:
        emit("worker.log", message=f"未知桌面端命令：{action}")


def main() -> int:
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    emit(
        "worker.ready",
        project_root=str(project_root()),
        algorithm_version="1.4.1",
        ffmpeg_path=bundled_tool("ffmpeg") or "PATH",
        ffprobe_path=bundled_tool("ffprobe") or "PATH",
    )
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            command = json.loads(line)
            if not isinstance(command, Mapping):
                raise ValueError("命令必须是 JSON 对象")
            handle_command(command)
        except Exception as exc:
            emit(
                "worker.error",
                error=str(exc),
                traceback=traceback.format_exc(),
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
