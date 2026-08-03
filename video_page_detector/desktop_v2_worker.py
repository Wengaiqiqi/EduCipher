from __future__ import annotations

import json
import os
import re
import shutil
import sys
import threading
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping

from .cloud_pipeline import CloudPagePipeline
from .config import DetectorConfig
from .llm_evaluation import (
    PROMPT_VERSION,
    LLMEvaluationConfig,
    evaluate_page,
    evaluate_transcript,
    render_evaluation_markdown,
    summarize_evaluations,
)
from .mimo_asr import (
    MimoASRSettings,
    resolve_mimo_api_key,
    transcribe_pages_with_mimo,
)
from .output_paths import resolve_run_directory, validate_video_id
from .pipeline import VideoPageDetector
from .transcription import (
    TranscriptionConfig,
    render_page_transcripts_markdown,
    transcribe_video_pages,
)
from 场景阈值方法.legacy_ffmpeg_scene_detector.config import (
    DetectorConfig as SceneThresholdDetectorConfig,
)
from 场景阈值方法.legacy_ffmpeg_scene_detector.pipeline import (
    VideoPageDetector as SceneThresholdVideoPageDetector,
)


_output_lock = threading.Lock()
_task_lock = threading.Lock()
_cancel_event = threading.Event()
_active_thread: threading.Thread | None = None
_active_task_dir: Path | None = None
_pending_retry_payload: dict[str, Any] | None = None


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


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_metadata_path(run_dir: Path) -> Path:
    return run_dir / "run_metadata.json"


def write_run_metadata(
    run_dir: Path,
    started_at: float,
    *,
    mode: str,
    include_llm: bool,
    detector_algorithm: str,
    video_path: str,
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
                    "detector_algorithm": detector_algorithm,
                    "video_path": video_path,
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


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".wmv", ".m4v", ".webm"}
VIDEO_SEARCH_SKIP_DIRS = {
    "$recycle.bin",
    "system volume information",
    "windows",
    "program files",
    "program files (x86)",
    "programdata",
    "node_modules",
    ".git",
    "appdata",
}


def discover_original_video(
    video_id: str,
    *,
    preferred_paths: list[str] | None = None,
    search_roots: list[Path] | None = None,
) -> Path | None:
    """Find an old task's source video without interrupting retry with a picker."""
    for raw in preferred_paths or []:
        if not raw:
            continue
        candidate = Path(raw).expanduser()
        if candidate.is_file():
            return candidate.resolve()

    roots: list[tuple[Path, int]] = []
    if search_roots is not None:
        roots.extend((Path(root), 6) for root in search_roots)
    else:
        drive_roots = [
            Path(f"{letter}:\\")
            for letter in "EFGHIJKLMNOPQRSTUVWXYZDABC"
            if Path(f"{letter}:\\").is_dir()
        ]
        roots.extend((root, 3) for root in drive_roots)
        home = Path.home()
        roots.extend(
            (home / name, 6)
            for name in ("Desktop", "Downloads", "Documents", "Videos")
        )

    expected_stem = video_id.casefold()
    visited: set[str] = set()
    for root, max_depth in roots:
        try:
            resolved_root = root.resolve()
        except OSError:
            continue
        marker = str(resolved_root).casefold()
        if marker in visited or not resolved_root.is_dir():
            continue
        visited.add(marker)
        queue: deque[tuple[Path, int]] = deque([(resolved_root, 0)])
        while queue:
            directory, depth = queue.popleft()
            try:
                entries = list(directory.iterdir())
            except (OSError, PermissionError):
                continue
            for entry in entries:
                try:
                    if entry.is_file():
                        if (
                            entry.stem.casefold() == expected_stem
                            and entry.suffix.casefold() in VIDEO_EXTENSIONS
                        ):
                            return entry.resolve()
                    elif (
                        depth < max_depth
                        and entry.name.casefold() not in VIDEO_SEARCH_SKIP_DIRS
                        and not entry.name.startswith(".")
                    ):
                        queue.append((entry, depth + 1))
                except (OSError, PermissionError):
                    continue
    return None


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
    metadata_status = str(metadata.get("status") or "")
    mode = str(
        metadata.get("mode")
        or ("full" if transcript or evaluation else "detect")
    )
    include_llm = bool(metadata.get("include_llm", bool(evaluation)))
    actively_running = bool(
        _active_thread is not None
        and _active_thread.is_alive()
        and _active_task_dir == run_dir.resolve()
    )
    interrupted = (
        metadata_status in {"running", "retrying"} and not actively_running
    ) or metadata_status in {"failed", "cancelled"}
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
            row["status"] = (
                "failed"
                if page.get("failure_stage") == "asr"
                or page.get("transcription_status") == "failed"
                else "detected"
            )
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
            if row["status"] == "failed" and not row.get("failure_stage"):
                row["failure_stage"] = "llm"
    if mode == "full" and interrupted:
        transcript_ids = {
            int(page["page_id"])
            for page in (transcript or {}).get("pages", [])
            if isinstance(page, Mapping) and "page_id" in page
        }
        evaluation_ids = {
            int(page["page_id"])
            for page in (evaluation or {}).get("pages", [])
            if isinstance(page, Mapping) and "page_id" in page
        }
        for page_id, row in page_map.items():
            if page_id not in transcript_ids:
                row.update(
                    {
                        "status": "failed",
                        "failure_stage": "asr",
                        "reason": "任务在本页语音识别完成前中断，可重试本页语音识别。",
                    }
                )
            elif include_llm and page_id not in evaluation_ids:
                row.update(
                    {
                        "status": "failed",
                        "failure_stage": "llm",
                        "reason": "任务在本页关联度评分完成前中断，可重试本页关联度评分。",
                    }
                )
    try:
        updated_at = (run_dir / "result.json").stat().st_mtime * 1000
    except OSError:
        updated_at = 0
    elapsed = load_run_elapsed(run_dir)
    evaluation_summary = evaluation.get("summary", {}) if evaluation else {}
    failed_pages = int(evaluation_summary.get("failed_pages") or 0)
    asr_failed_pages = int(
        (
            transcript.get("transcription", {})
            .get("cloud_statistics", {})
            .get("failed_page_count", 0)
        )
        if transcript
        else 0
    )
    evaluation_complete = evaluation_summary.get("complete")
    recovered_failed_pages = sum(
        page.get("status") == "failed" for page in page_map.values()
    )
    completed_with_errors = bool(
        evaluation
        and (failed_pages > 0 or evaluation_complete is False)
    ) or asr_failed_pages > 0 or recovered_failed_pages > 0 or metadata_status == "completed_with_errors"
    # Finalization calls load_task() before the worker thread exits.  A persisted
    # terminal status must therefore win over the still-alive thread; otherwise
    # task.completed carries status="running" and the desktop stays stuck forever.
    if metadata_status == "completed_with_errors":
        task_status = "completed_with_errors"
    elif metadata_status == "completed":
        task_status = "completed"
    elif actively_running:
        task_status = "running"
    elif completed_with_errors:
        task_status = "completed_with_errors"
    elif evaluation:
        task_status = "completed"
    elif metadata_status in {"failed", "cancelled"}:
        task_status = "failed"
    else:
        task_status = "idle"
    finished = task_status in {"completed", "completed_with_errors"}
    total_pages = max(len(page_map), 1)
    transcript_page_count = len({
        int(page["page_id"])
        for page in (transcript or {}).get("pages", [])
        if isinstance(page, Mapping) and "page_id" in page
    })
    evaluation_page_count = len({
        int(page["page_id"])
        for page in (evaluation or {}).get("pages", [])
        if isinstance(page, Mapping) and "page_id" in page
    })
    stage_progresses = {
        "ppt": 100,
        "voice": (
            0
            if mode == "detect"
            else min(100, round(transcript_page_count / total_pages * 100))
        ),
        "llm": (
            0
            if not include_llm
            else min(100, round(evaluation_page_count / total_pages * 100))
        ),
        "report": 100 if finished else 0,
    }
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
        "video_path": str(
            detection.get("video_path") or metadata.get("video_path") or ""
        ),
        "run_dir": str(run_dir.resolve()),
        "updated_at": updated_at,
        "status": task_status,
        "progress": 100 if finished else (78 if transcript else 36),
        "stage": (
            "正在处理"
            if task_status == "running"
            else (
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
            )
        ),
        "elapsed_sec": elapsed,
        "model": str(evaluation.get("model") or "") if evaluation else "",
        "include_evidence": evidence_enabled,
        "include_llm": include_llm,
        "mode": mode,
        "completed_stages": completed_stages,
        "stage_progresses": stage_progresses,
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


def task_roots(output_root: str | None = None) -> list[Path]:
    candidates = [
        Path(output_root) if output_root else None,
        project_root() / "output",
        Path.home() / "Documents" / "课堂PPT处理结果",
    ]
    roots: list[Path] = []
    for candidate in candidates:
        if candidate is None:
            continue
        resolved = candidate.expanduser().resolve()
        if resolved not in roots:
            roots.append(resolved)
    return roots


def resolve_existing_task_dir(
    task_id: str,
    output_root: str | None = None,
) -> Path:
    candidate = Path(task_id).expanduser().resolve()
    if not any(candidate.parent == root for root in task_roots(output_root)):
        raise ValueError("任务目录不在允许的结果目录中。")
    if not candidate.is_dir() or load_task(candidate) is None:
        raise FileNotFoundError("任务结果不存在或已经被删除。")
    return candidate


def delete_task_result(task_id: str, output_root: str | None = None) -> Path:
    run_dir = resolve_existing_task_dir(task_id, output_root)
    with _task_lock:
        is_actively_running = bool(
            _active_thread is not None
            and _active_thread.is_alive()
            and _active_task_dir == run_dir
        )
    if is_actively_running:
        raise RuntimeError("正在处理或重试的任务不能删除。")
    shutil.rmtree(run_dir)
    return run_dir


def _write_retry_evaluation(
    *,
    run_dir: Path,
    transcript: Mapping[str, Any],
    previous: Mapping[str, Any],
    config: LLMEvaluationConfig,
    pages: list[dict[str, Any]],
    retry_elapsed: float,
) -> dict[str, Any]:
    destination = run_dir / "llm_evaluation"
    result_path = destination / "llm_evaluation.json"
    artifacts = dict(previous.get("artifacts", {}))
    report_path = Path(
        str(artifacts.get("report_markdown") or destination / "PPT讲话关联度报告.md")
    )
    artifacts.update(
        {
            "result_json": result_path.resolve().as_posix(),
            "report_markdown": report_path.resolve().as_posix(),
            "page_results_dir": (destination / "pages").resolve().as_posix(),
        }
    )
    payload: dict[str, Any] = {
        **dict(previous),
        "video_id": str(transcript.get("video_id") or run_dir.name),
        "transcript_path": (run_dir / "transcript.json").resolve().as_posix(),
        "model": config.model,
        "base_url": config.base_url,
        "prompt_version": PROMPT_VERSION,
        "processing_duration_sec": round(
            float(previous.get("processing_duration_sec") or 0) + retry_elapsed,
            3,
        ),
        "summary": summarize_evaluations(pages),
        "pages": pages,
        "config": {
            key: value
            for key, value in asdict(config).items()
            if key != "api_key_env"
        },
        "artifacts": artifacts,
    }
    write_json(result_path, payload)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_evaluation_markdown(payload), encoding="utf-8")
    return payload


def run_retry_failed_pages(payload: Mapping[str, Any]) -> None:
    global _active_task_dir
    started_at = time.perf_counter()
    task_id = str(payload.get("task_id") or "")
    run_dir: Path | None = None
    llm_executor: ThreadPoolExecutor | None = None
    previous_metadata_status = "completed_with_errors"
    try:
        run_dir = resolve_existing_task_dir(
            task_id,
            str(payload.get("output_root") or "") or None,
        )
        with _task_lock:
            _active_task_dir = run_dir
        task_id = str(run_dir.resolve())
        detection = read_json(run_dir / "result.json") or {}
        transcript_path = run_dir / "transcript.json"
        transcript = read_json(transcript_path)
        evaluation_path = run_dir / "llm_evaluation" / "llm_evaluation.json"
        evaluation = read_json(evaluation_path) or {}
        metadata = read_json(run_metadata_path(run_dir)) or {}
        previous_metadata_status = str(
            metadata.get("status") or "completed_with_errors"
        )
        interrupted = str(metadata.get("status") or "") in {
            "running",
            "retrying",
            "failed",
            "cancelled",
        }
        if not transcript or not isinstance(transcript.get("pages"), list):
            if not interrupted or not isinstance(detection.get("pages"), list):
                raise ValueError("该任务没有可供单页重试的转写结果。")
            transcript = {
                "video_id": str(detection.get("video_id") or run_dir.name),
                "video_path": str(
                    payload.get("video_path")
                    or metadata.get("video_path")
                    or detection.get("video_path")
                    or ""
                ),
                "video_duration_sec": detection.get("video_duration_sec"),
                "ppt_result_path": str((run_dir / "result.json").resolve()),
                "pages": [
                    {
                        **dict(page),
                        "speech_text": "",
                        "utterances": [],
                        "transcription_status": "failed",
                        "failure_stage": "asr",
                        "reason": "任务在本页语音识别完成前中断。",
                    }
                    for page in detection["pages"]
                    if isinstance(page, Mapping) and "page_id" in page
                ],
                "transcription": {
                    "engine": "mimo-cloud",
                    "cloud_statistics": {
                        "failed_page_count": len(detection["pages"]),
                    },
                },
                "artifacts": {
                    "transcript_json": str(transcript_path.resolve()),
                    "page_transcript_markdown": str(
                        (run_dir / "逐页语音文字.md").resolve()
                    ),
                },
            }
            write_json(transcript_path, transcript)
        elif interrupted:
            transcript_by_id = {
                int(page["page_id"]): dict(page)
                for page in transcript.get("pages", [])
                if isinstance(page, Mapping) and "page_id" in page
            }
            for page in detection.get("pages", []):
                if not isinstance(page, Mapping) or "page_id" not in page:
                    continue
                page_id = int(page["page_id"])
                if page_id not in transcript_by_id:
                    transcript_by_id[page_id] = {
                        **dict(page),
                        "speech_text": "",
                        "utterances": [],
                        "transcription_status": "failed",
                        "failure_stage": "asr",
                        "reason": "任务在本页语音识别完成前中断。",
                    }
            transcript["pages"] = [
                transcript_by_id[key] for key in sorted(transcript_by_id)
            ]
        if interrupted and bool(metadata.get("include_llm", True)) and not evaluation:
            evaluation = {
                "video_id": str(transcript.get("video_id") or run_dir.name),
                "pages": [],
                "summary": {"failed_pages": 0, "complete": False},
            }

        failed_by_id: dict[int, dict[str, Any]] = {}
        for page in transcript.get("pages", []):
            if (
                isinstance(page, Mapping)
                and page.get("failure_stage") == "asr"
                and "page_id" in page
            ):
                failed_by_id[int(page["page_id"])] = dict(page)
        for item in evaluation.get("pages", []):
            if not isinstance(item, Mapping) or item.get("status") != "failed":
                continue
            page_id = int(item["page_id"])
            merged = dict(failed_by_id.get(page_id, {}))
            merged.update(dict(item))
            merged.setdefault("failure_stage", "llm")
            failed_by_id[page_id] = merged
        if interrupted and bool(metadata.get("include_llm", True)):
            evaluated_ids = {
                int(item["page_id"])
                for item in evaluation.get("pages", [])
                if isinstance(item, Mapping) and "page_id" in item
            }
            for page in transcript.get("pages", []):
                if (
                    isinstance(page, Mapping)
                    and "page_id" in page
                    and page.get("failure_stage") != "asr"
                    and int(page["page_id"]) not in evaluated_ids
                ):
                    failed_by_id[int(page["page_id"])] = {
                        **dict(page),
                        "failure_stage": "llm",
                        "reason": "任务在本页关联度评分完成前中断。",
                    }

        requested = payload.get("page_ids")
        if isinstance(requested, list) and requested:
            target_ids = sorted({int(value) for value in requested})
            invalid = [page_id for page_id in target_ids if page_id not in failed_by_id]
            if invalid:
                raise ValueError(f"以下页面当前不是失败状态：{invalid}")
        else:
            target_ids = sorted(failed_by_id)
        if not target_ids:
            raise ValueError("该任务目前没有需要重试的失败页面。")

        settings = payload.get("settings", {})
        if not isinstance(settings, Mapping):
            raise ValueError("桌面端设置格式不正确。")
        metadata["status"] = "retrying"
        write_json(run_metadata_path(run_dir), metadata)

        transcript_pages = {
            int(page["page_id"]): dict(page)
            for page in transcript["pages"]
            if isinstance(page, Mapping) and "page_id" in page
        }
        asr_target_ids = [
            page_id
            for page_id in target_ids
            if failed_by_id[page_id].get("failure_stage") == "asr"
        ]
        llm_target_ids = [
            page_id
            for page_id in target_ids
            if failed_by_id[page_id].get("failure_stage") != "asr"
        ]
        retry_asr_config = (
            transcription_config(settings) if asr_target_ids else None
        )
        will_retry_llm = bool(
            llm_target_ids or (asr_target_ids and evaluation)
        )
        retry_llm_settings = llm_config(settings) if will_retry_llm else None
        emit(
            "task.retry_started",
            task_id=task_id,
            page_ids=target_ids,
            asr_concurrency=(
                retry_asr_config.mimo_max_concurrency
                if retry_asr_config is not None
                else 0
            ),
            llm_concurrency=(
                retry_llm_settings.max_concurrency
                if retry_llm_settings is not None
                else 0
            ),
        )
        activity_lock = threading.Lock()
        active_asr = 0
        active_llm = 0
        peak_asr = 0
        peak_llm = 0
        retry_cloud_limit = min(
            10,
            (retry_asr_config.mimo_max_concurrency if retry_asr_config else 0)
            + (retry_llm_settings.max_concurrency if retry_llm_settings else 0),
        )
        retry_cloud_slots = threading.BoundedSemaphore(max(1, retry_cloud_limit))

        def emit_retry_activity() -> None:
            emit(
                "cloud.activity",
                task_id=task_id,
                active_cloud_requests=active_asr + active_llm,
                asr_active_requests=active_asr,
                llm_active_requests=active_llm,
                cloud_limit=retry_cloud_limit,
            )

        evaluation_pages: dict[int, dict[str, Any]] = {}
        llm_futures: dict[Any, int] = {}
        llm_submitted_ids: set[int] = set()
        llm_settings = retry_llm_settings
        llm_key = ""

        if llm_settings is not None:
            if not bool(payload.get("llm_upload_consent")):
                raise ValueError("请确认允许重新发送失败页截图和课堂转写。")
            llm_key = str(payload.get("llm_api_key") or "").strip()
            llm_key = llm_key or os.environ.get("LLM_API_KEY", "").strip()
            if not llm_key:
                raise ValueError("没有找到 LLM API Key。")
            evaluation_pages = {
                int(item["page_id"]): dict(item)
                for item in evaluation.get("pages", [])
                if isinstance(item, Mapping) and "page_id" in item
            }

            def retry_llm(page_id: int) -> tuple[int, dict[str, Any]]:
                nonlocal active_llm, peak_llm
                require_not_cancelled()
                active_page = dict(transcript_pages[page_id])
                active_page.pop("failure_stage", None)
                active_page.pop("reason", None)
                with retry_cloud_slots:
                    emit_page(active_page, "scoring", task_id=task_id)
                    with activity_lock:
                        active_llm += 1
                        peak_llm = max(peak_llm, active_llm)
                        emit_retry_activity()
                    try:
                        try:
                            result = evaluate_page(
                                transcript_pages[page_id],
                                transcript_path=transcript_path,
                                config=llm_settings,
                                output_dir=run_dir / "llm_evaluation",
                                api_key=llm_key,
                            )
                        except Exception as exc:
                            result = {
                                "page_id": page_id,
                                "start_sec": transcript_pages[page_id].get(
                                    "start_sec", 0
                                ),
                                "end_sec": transcript_pages[page_id].get(
                                    "end_sec", 0
                                ),
                                "status": "failed",
                                "failure_stage": "llm",
                                "speech_relevance": 0,
                                "ppt_coverage": 0,
                                "evidence_consistency": 0,
                                "score": 0,
                                "level": "请求失败",
                                "reason": str(exc),
                            }
                        if result.get("status") == "failed":
                            result["failure_stage"] = "llm"
                        kwargs = {
                            "score": result.get("score"),
                            "level": result.get("level"),
                            "reason": result.get("reason"),
                            "failure_stage": result.get("failure_stage"),
                        }
                        if "matched_evidence" in result:
                            kwargs["evidence"] = result.get("matched_evidence")
                        emit_page(
                            transcript_pages[page_id],
                            (
                                "failed"
                                if result.get("status") == "failed"
                                else "completed"
                            ),
                            task_id=task_id,
                            **kwargs,
                        )
                        return page_id, result
                    finally:
                        with activity_lock:
                            active_llm = max(0, active_llm - 1)
                            emit_retry_activity()

            total_llm_targets = len(
                set(llm_target_ids)
                | (set(asr_target_ids) if evaluation else set())
            )
            llm_executor = ThreadPoolExecutor(
                max_workers=min(llm_settings.max_concurrency, total_llm_targets),
                thread_name_prefix="retry-page-llm",
            )

            def submit_llm_retry(page_id: int) -> None:
                if page_id in llm_submitted_ids:
                    return
                llm_submitted_ids.add(page_id)
                assert llm_executor is not None
                llm_futures[llm_executor.submit(retry_llm, page_id)] = page_id

            for page_id in llm_target_ids:
                submit_llm_retry(page_id)

        if asr_target_ids:
            if not bool(payload.get("asr_upload_consent")):
                raise ValueError("请确认允许重新发送失败页的临时音频。")
            assert retry_asr_config is not None
            asr_config = retry_asr_config
            if asr_config.engine != "mimo-cloud":
                raise ValueError("云端 ASR 失败页重试需要选择 ASR 模型服务。")
            asr_key = resolve_mimo_api_key(
                asr_config.mimo_api_key_env,
                str(payload.get("asr_api_key") or ""),
            )
            video_path = discover_original_video(
                str(
                    detection.get("video_id")
                    or transcript.get("video_id")
                    or run_dir.name
                ),
                preferred_paths=[
                    str(payload.get("video_path") or ""),
                    str(transcript.get("video_path") or ""),
                    str(metadata.get("video_path") or ""),
                    str(detection.get("video_path") or ""),
                ],
            )
            if video_path is None:
                raise FileNotFoundError(
                    "未能根据任务名称自动找到原始视频，无法重试语音识别。"
                )
            transcript["video_path"] = str(video_path.resolve())
            metadata["video_path"] = str(video_path.resolve())
            detection["video_path"] = str(video_path.resolve())
            write_json(run_dir / "result.json", detection)
            write_json(run_metadata_path(run_dir), metadata)
            asr_settings = MimoASRSettings(
                base_url=asr_config.mimo_base_url,
                model=asr_config.mimo_model,
                language=asr_config.mimo_language,
                max_concurrency=1,
                max_chunk_duration_sec=asr_config.mimo_max_chunk_duration_sec,
                timeout_sec=asr_config.mimo_timeout_sec,
                max_retries=asr_config.mimo_max_retries,
                ffmpeg_path=asr_config.ffmpeg_path,
            )
            def retry_asr(page_id: int) -> tuple[int, dict[str, Any]]:
                nonlocal active_asr, peak_asr
                require_not_cancelled()
                active_page = dict(transcript_pages[page_id])
                active_page.pop("failure_stage", None)
                active_page.pop("reason", None)
                with retry_cloud_slots:
                    emit_page(active_page, "transcribing", task_id=task_id)
                    with activity_lock:
                        active_asr += 1
                        peak_asr = max(peak_asr, active_asr)
                        emit_retry_activity()
                    try:
                        pages, _ = transcribe_pages_with_mimo(
                            video_path,
                            [transcript_pages[page_id]],
                            settings=asr_settings,
                            api_key=asr_key,
                        )
                        return page_id, pages[0]
                    finally:
                        with activity_lock:
                            active_asr = max(0, active_asr - 1)
                            emit_retry_activity()

            with ThreadPoolExecutor(
                max_workers=min(asr_config.mimo_max_concurrency, len(asr_target_ids)),
                thread_name_prefix="retry-page-asr",
            ) as executor:
                futures = {
                    executor.submit(retry_asr, page_id): page_id
                    for page_id in asr_target_ids
                }
                for future in as_completed(futures):
                    page_id = futures[future]
                    try:
                        _, result = future.result()
                        result.pop("failure_stage", None)
                        result.pop("reason", None)
                        result["transcription_status"] = "completed"
                        transcript_pages[page_id] = result
                        if evaluation:
                            submit_llm_retry(page_id)
                        else:
                            emit_page(result, "completed", task_id=task_id)
                    except Exception as exc:
                        failed = transcript_pages[page_id]
                        failed.update(
                            {
                                "speech_text": "",
                                "utterances": [],
                                "transcription_status": "failed",
                                "failure_stage": "asr",
                                "reason": str(exc),
                            }
                        )
                        emit_page(
                            failed,
                            "failed",
                            task_id=task_id,
                            failure_stage="asr",
                            reason=str(exc),
                        )

            ordered_transcripts = [
                transcript_pages[key] for key in sorted(transcript_pages)
            ]
            transcript["pages"] = ordered_transcripts
            transcript["config"] = asdict(asr_config)
            transcript.setdefault("transcription", {})["model"] = asr_config.mimo_model
            transcript["transcription"]["utterance_count"] = sum(
                len(page.get("utterances", [])) for page in ordered_transcripts
            )
            cloud_statistics = transcript["transcription"].setdefault(
                "cloud_statistics", {}
            )
            cloud_statistics["failed_page_count"] = sum(
                page.get("failure_stage") == "asr" for page in ordered_transcripts
            )
            cloud_statistics["retry_request_count"] = int(
                cloud_statistics.get("retry_request_count", 0)
            ) + len(asr_target_ids)
            cloud_statistics["retry_max_concurrency"] = asr_config.mimo_max_concurrency
            cloud_statistics["retry_peak_concurrency"] = peak_asr
            write_json(transcript_path, transcript)
            transcript_markdown = Path(
                str(
                    transcript.get("artifacts", {}).get("page_transcript_markdown")
                    or run_dir / "逐页语音文字.md"
                )
            )
            transcript_markdown.write_text(
                render_page_transcripts_markdown(
                    str(transcript.get("video_id") or run_dir.name),
                    ordered_transcripts,
                ),
                encoding="utf-8",
            )

        if llm_executor is not None:
            for future in as_completed(list(llm_futures)):
                page_id = llm_futures[future]
                try:
                    _, result = future.result()
                except Exception as exc:
                    result = {
                        "page_id": page_id,
                        "start_sec": transcript_pages[page_id].get("start_sec", 0),
                        "end_sec": transcript_pages[page_id].get("end_sec", 0),
                        "status": "failed",
                        "failure_stage": "llm",
                        "speech_relevance": 0,
                        "ppt_coverage": 0,
                        "evidence_consistency": 0,
                        "score": 0,
                        "level": "请求失败",
                        "reason": str(exc),
                    }
                    emit_page(
                        transcript_pages[page_id],
                        "failed",
                        task_id=task_id,
                        failure_stage="llm",
                        reason=str(exc),
                    )
                evaluation_pages[page_id] = result
            llm_executor.shutdown(wait=True)
            llm_executor = None
        if llm_futures:
            assert llm_settings is not None
            ordered_evaluations = [
                evaluation_pages[key] for key in sorted(evaluation_pages)
            ]
            evaluation = _write_retry_evaluation(
                run_dir=run_dir,
                transcript=transcript,
                previous=evaluation,
                config=llm_settings,
                pages=ordered_evaluations,
                retry_elapsed=time.perf_counter() - started_at,
            )

        retry_elapsed = round(time.perf_counter() - started_at, 3)
        metadata = read_json(run_metadata_path(run_dir)) or metadata
        metadata["elapsed_sec"] = round(
            float(metadata.get("elapsed_sec") or 0) + retry_elapsed,
            3,
        )
        metadata["completed_at"] = time.time()
        metadata["retry_asr_concurrency"] = (
            retry_asr_config.mimo_max_concurrency if retry_asr_config else 0
        )
        metadata["retry_llm_concurrency"] = (
            retry_llm_settings.max_concurrency if retry_llm_settings else 0
        )
        metadata["retry_peak_asr_concurrency"] = peak_asr
        metadata["retry_peak_llm_concurrency"] = peak_llm
        asr_failed = sum(
            page.get("failure_stage") == "asr"
            for page in transcript.get("pages", [])
            if isinstance(page, Mapping)
        )
        llm_failed = int(
            evaluation.get("summary", {}).get("failed_pages", 0)
            if evaluation
            else 0
        )
        metadata["status"] = (
            "completed_with_errors" if asr_failed or llm_failed else "completed"
        )
        write_json(run_metadata_path(run_dir), metadata)
        task = load_task(run_dir)
        emit(
            "task.retry_completed",
            task_id=task_id,
            page_ids=target_ids,
            result=task,
        )
    except Exception as exc:
        if run_dir is not None:
            metadata = read_json(run_metadata_path(run_dir)) or {}
            metadata["status"] = previous_metadata_status
            write_json(run_metadata_path(run_dir), metadata)
        emit(
            "task.retry_failed",
            task_id=task_id,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
    finally:
        if llm_executor is not None:
            llm_executor.shutdown(wait=False, cancel_futures=True)
        with _task_lock:
            if run_dir is not None and _active_task_dir == run_dir.resolve():
                _active_task_dir = None
        _cancel_event.clear()
        launch_pending_retry()


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
    config = replace(
        config,
        ffmpeg_path=bundled_tool("ffmpeg") or config.ffmpeg_path,
        ffprobe_path=bundled_tool("ffprobe") or config.ffprobe_path,
    )
    config.validate()
    return config


def detector_algorithm(settings: Mapping[str, Any]) -> str:
    value = str(settings.get("detector_algorithm") or "temporal").strip()
    if value not in {"temporal", "scene-threshold"}:
        raise ValueError(f"不支持的 PPT 页面识别算法：{value}")
    return value


def scene_threshold_detector_config() -> SceneThresholdDetectorConfig:
    config = SceneThresholdDetectorConfig()
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
    engine = str(settings.get("asr_engine") or base.engine)
    model = str(settings.get("asr_model") or base.model).strip()
    if settings.get("asr_engine") == "faster-whisper":
        if not str(settings.get("asr_model") or "").strip():
            raise ValueError("请选择本地 faster-whisper 模型权重文件夹。")
        model_dir = Path(model).expanduser()
        if not model_dir.is_dir():
            raise ValueError(f"本地模型文件夹不存在：{model_dir}")
        if not (model_dir / "model.bin").is_file():
            raise ValueError(
                f"所选文件夹中没有 model.bin，请选择完整的 faster-whisper 模型目录：{model_dir}"
            )
        model = str(model_dir.resolve())
    config = replace(
        base,
        engine=engine,
        model=model,
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
    global _active_task_dir
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
        with _task_lock:
            _active_task_dir = run_dir.resolve()
        result_path = run_dir / "result.json"
        transcript_path = run_dir / "transcript.json"
        evaluation_dir = run_dir / "llm_evaluation"
        task_id = str(run_dir.resolve())
        include_llm = bool(settings.get("include_llm", True)) and mode == "full"

        selected_detector_algorithm = detector_algorithm(settings)
        emit(
            "task.started",
            task_id=task_id,
            video_id=video_id,
            video_path=str(video.resolve()),
            run_dir=str(run_dir.resolve()),
            started_at=time.time(),
            algorithm_version="1.4.11",
            streaming_page_confirmation=(selected_detector_algorithm == "temporal"),
            detector_algorithm=selected_detector_algorithm,
            mode=mode,
            include_llm=include_llm,
            include_evidence=bool(settings.get("include_evidence", False)),
        )
        detector = (
            VideoPageDetector(default_detector_config(settings))
            if selected_detector_algorithm == "temporal"
            else SceneThresholdVideoPageDetector(scene_threshold_detector_config())
        )
        write_run_metadata(
            run_dir,
            time.time(),
            mode=mode,
            include_llm=include_llm,
            detector_algorithm=selected_detector_algorithm,
            video_path=str(video.resolve()),
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
                        emit_page(
                            page,
                            "cloud_waiting" if include_llm else "completed",
                            task_id=task_id,
                        )
                    elif page_id in cloud_pipeline._asr_errors:
                        failed_page = cloud_pipeline._submitted_pages.get(
                            page_id, {"page_id": page_id}
                        )
                        emit_page(
                            failed_page,
                            "failed",
                            task_id=task_id,
                            failure_stage="asr",
                            reason=str(cloud_pipeline._asr_errors[page_id]),
                        )
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
                        failed = evaluation.get("status") == "failed"
                        if failed:
                            kwargs["failure_stage"] = "llm"
                        matched = evaluation.get("matched_evidence")
                        if matched is not None:
                            kwargs["evidence"] = matched
                        emit_page(
                            transcript,
                            "failed" if failed else "completed",
                            task_id=task_id,
                            **kwargs,
                        )

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
                page_activity_callback=lambda page, status: emit_page(
                    page,
                    status,
                    task_id=task_id,
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
                "cloud_waiting" if cloud_pipeline else "detected",
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

        if selected_detector_algorithm == "temporal":
            detection = detector.run(
                video,
                output_root=output_root,
                video_id=video_id,
                progress_callback=detection_progress,
                page_ready_callback=page_ready,
            )
        else:
            detection = detector.run(
                video,
                output_root=output_root,
                video_id=video_id,
                progress_callback=detection_progress,
            )
            pages = detection.get("pages", [])
            for page_index, page in enumerate(pages, start=1):
                if isinstance(page, dict):
                    page_ready(page, page_index, len(pages))
        analysis = detection.setdefault("analysis", {})
        if isinstance(analysis, dict):
            analysis["detector_algorithm"] = selected_detector_algorithm
        detection["video_path"] = str(video.resolve())
        write_json(result_path, detection)
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
            asr_failed_pages = int(
                (
                    transcript.get("transcription", {})
                    .get("cloud_statistics", {})
                    .get("failed_page_count", 0)
                )
                if transcript
                else 0
            )
            meta["status"] = (
                "completed_with_errors"
                if asr_failed_pages > 0
                or (
                    evaluation
                    and (
                        int(evaluation_summary.get("failed_pages") or 0) > 0
                        or evaluation_summary.get("complete") is False
                    )
                )
                else "completed"
            )
            run_metadata_path(run_dir).write_text(
                json.dumps(meta, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except OSError:
            pass
        emit(
            "task.progress",
            task_id=task_id,
            stage="生成报告",
            message="分析报告已生成，正在保存任务状态",
            progress=99,
            stage_progress=100,
            completed_stage="report",
        )
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
        with _task_lock:
            if run_dir is not None and _active_task_dir == run_dir.resolve():
                _active_task_dir = None
        _cancel_event.clear()
        launch_pending_retry()


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


def start_retry_failed_pages(payload: Mapping[str, Any]) -> None:
    global _active_thread, _pending_retry_payload
    with _task_lock:
        if _active_thread is not None and _active_thread.is_alive():
            queued = dict(payload)
            if _pending_retry_payload is not None:
                existing_ids = _pending_retry_payload.get("page_ids")
                incoming_ids = queued.get("page_ids")
                if isinstance(existing_ids, list) and isinstance(incoming_ids, list):
                    queued["page_ids"] = sorted(
                        {int(value) for value in existing_ids + incoming_ids}
                    )
                else:
                    queued.pop("page_ids", None)
            _pending_retry_payload = queued
            emit(
                "task.retry_queued",
                task_id=str(payload.get("task_id") or ""),
                page_ids=queued.get("page_ids") or [],
                message="当前主流程仍在处理，失败页重试已加入队列。",
            )
            return
        _cancel_event.clear()
        _active_thread = threading.Thread(
            target=run_retry_failed_pages,
            args=(dict(payload),),
            daemon=True,
            name="desktop-v2-page-retry",
        )
        _active_thread.start()


def launch_pending_retry() -> None:
    global _active_thread, _pending_retry_payload
    with _task_lock:
        if _pending_retry_payload is None:
            return
        payload = _pending_retry_payload
        _pending_retry_payload = None
        _active_thread = threading.Thread(
            target=run_retry_failed_pages,
            args=(payload,),
            daemon=True,
            name="desktop-v2-page-retry",
        )
        _active_thread.start()


def handle_command(command: Mapping[str, Any]) -> None:
    action = str(command.get("action") or "")
    if action == "ping":
        emit(
            "worker.ready",
            project_root=str(project_root()),
            algorithm_version="1.4.11",
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
    elif action == "delete_task":
        task_id = str(command.get("task_id") or "")
        try:
            delete_task_result(
                task_id,
                str(command.get("output_root") or "") or None,
            )
            emit("task.deleted", task_id=task_id)
        except Exception as exc:
            emit("task.delete_failed", task_id=task_id, error=str(exc))
    elif action == "retry_failed_pages":
        payload = command.get("payload", {})
        if not isinstance(payload, Mapping):
            emit("task.retry_failed", error="重试参数格式不正确。")
            return
        start_retry_failed_pages(payload)
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
        algorithm_version="1.4.11",
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
