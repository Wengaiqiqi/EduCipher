from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import traceback
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from .cloud_pipeline import CloudPagePipeline
from .config import DetectorConfig
from .llm_evaluation import LLMEvaluationConfig, evaluate_transcript
from .output_paths import resolve_run_directory, validate_video_id
from .pipeline import VideoPageDetector
from .transcription import (
    TranscriptionConfig,
    format_timestamp,
    transcribe_video_pages,
)
from .mimo_asr import resolve_mimo_api_key


APP_NAME = "课堂PPT智能处理"
APP_VERSION = "1.4.1"
LOCAL_ASR_LABEL = "本地 faster-whisper"
MIMO_ASR_LABEL = "小米 MiMo 云端（推荐加速）"
VIDEO_FILE_TYPES = [
    ("视频文件", "*.mp4 *.mkv *.mov *.avi *.wmv *.m4v *.webm"),
    ("所有文件", "*.*"),
]


def application_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def resource_path(relative: str | Path) -> Path:
    relative_path = Path(relative)
    candidates: list[Path] = []
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        candidates.append(Path(bundle_root) / relative_path)
    candidates.extend(
        [
            application_dir() / relative_path,
            Path(__file__).resolve().parents[1] / relative_path,
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else relative_path


def settings_path() -> Path:
    base = Path(os.environ.get("APPDATA", Path.home()))
    return base / "ClassroomPPTProcessor" / "settings.json"


def default_output_root() -> Path:
    documents = Path.home() / "Documents"
    return (documents if documents.exists() else Path.home()) / "课堂PPT处理结果"


def open_external(path: Path) -> None:
    target = str(path.resolve())
    if sys.platform == "win32":
        os.startfile(target)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", target])
    else:
        subprocess.Popen(["xdg-open", target])


def sanitize_video_id(raw: str, fallback: str) -> str:
    value = raw.strip() or fallback.strip()
    try:
        return validate_video_id(value)
    except ValueError as exc:
        raise ValueError(f"视频名称无效：{exc}") from exc


@dataclass(frozen=True)
class WorkflowPaths:
    output_root: Path
    video_id: str
    run_dir: Path
    result_json: Path
    transcript_json: Path
    evaluation_dir: Path


def build_workflow_paths(
    output_root: str | Path,
    video_id: str,
) -> WorkflowPaths:
    root = Path(output_root).expanduser().resolve()
    run_dir = resolve_run_directory(root, video_id)
    return WorkflowPaths(
        output_root=root,
        video_id=video_id,
        run_dir=run_dir,
        result_json=run_dir / "result.json",
        transcript_json=run_dir / "transcript.json",
        evaluation_dir=run_dir / "llm_evaluation",
    )


def detector_config_for_preset(
    preset: str,
    *,
    ffmpeg_path: str | None = None,
    ffprobe_path: str | None = None,
) -> DetectorConfig:
    config = DetectorConfig()
    if preset == "快速预览":
        config = replace(
            config,
            temporal_sample_interval_sec=4.0,
            temporal_confirmation_sec=12.0,
            temporal_analysis_width=256,
            temporal_analysis_height=144,
            temporal_refinement_fps=2.0,
            jpeg_quality=85,
        )
    elif preset != "智能精准（推荐）":
        raise ValueError(f"未知处理模式：{preset}")
    config = replace(
        config,
        ffmpeg_path=ffmpeg_path,
        ffprobe_path=ffprobe_path,
    )
    config.validate()
    return config


def combine_page_rows(
    detection: Mapping[str, Any] | None,
    transcript: Mapping[str, Any] | None,
    evaluation: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    page_map: dict[int, dict[str, Any]] = {}
    for source in (detection, transcript):
        if not source:
            continue
        pages = source.get("pages", [])
        if not isinstance(pages, list):
            continue
        for page in pages:
            if not isinstance(page, Mapping) or "page_id" not in page:
                continue
            page_id = int(page["page_id"])
            row = page_map.setdefault(page_id, {"page_id": page_id})
            row.update(
                {
                    key: value
                    for key, value in page.items()
                    if key
                    in {
                        "start_sec",
                        "end_sec",
                        "screenshot_path",
                        "confidence",
                        "utterances",
                        "speech_text",
                    }
                }
            )
    if evaluation:
        pages = evaluation.get("pages", [])
        if isinstance(pages, list):
            for page in pages:
                if not isinstance(page, Mapping) or "page_id" not in page:
                    continue
                page_id = int(page["page_id"])
                row = page_map.setdefault(page_id, {"page_id": page_id})
                row.update(
                    {
                        "evaluation_status": page.get("status", ""),
                        "score": page.get("score", ""),
                        "level": page.get("level", ""),
                    }
                )
    rows: list[dict[str, Any]] = []
    for page_id in sorted(page_map):
        row = page_map[page_id]
        utterances = row.get("utterances", [])
        row["utterance_count"] = (
            len(utterances) if isinstance(utterances, list) else 0
        )
        rows.append(row)
    return rows


def run_packaged_self_test(destination: str | Path) -> dict[str, Any]:
    started_at = time.perf_counter()
    checks: dict[str, Any] = {}
    try:
        import av
        import ctranslate2
        import faster_whisper
        import httpx

        checks["imports"] = {
            "av": av.__version__,
            "ctranslate2": ctranslate2.__version__,
            "faster_whisper": getattr(faster_whisper, "__version__", "ok"),
            "httpx": httpx.__version__,
        }
        ffmpeg = resource_path("tools/ffmpeg.exe")
        ffprobe = resource_path("tools/ffprobe.exe")
        model_root = resource_path("models/faster-whisper")
        checks["resources"] = {
            "ffmpeg": ffmpeg.is_file(),
            "ffprobe": ffprobe.is_file(),
            "model_root": model_root.is_dir(),
            "model_bin_count": len(list(model_root.rglob("model.bin"))),
        }
        if not all(
            (
                checks["resources"]["ffmpeg"],
                checks["resources"]["ffprobe"],
                checks["resources"]["model_root"],
                checks["resources"]["model_bin_count"] > 0,
            )
        ):
            raise RuntimeError("发布资源不完整")
        version_result = subprocess.run(
            [str(ffmpeg), "-version"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=(
                subprocess.CREATE_NO_WINDOW
                if sys.platform == "win32"
                else 0
            ),
        )
        checks["ffmpeg_version"] = version_result.stdout.splitlines()[0]
        transcription_config = TranscriptionConfig.from_file(
            resource_path("config/transcription.json")
        )
        llm_config = LLMEvaluationConfig.from_file(
            resource_path("config/llm_evaluation.json")
        )
        checks["configs"] = {
            "transcription_model": transcription_config.model,
            "llm_model": llm_config.model,
        }
        from faster_whisper import WhisperModel

        model = WhisperModel(
            transcription_config.model,
            device=transcription_config.device,
            compute_type=transcription_config.compute_type,
            download_root=str(model_root),
        )
        checks["whisper_model_loaded"] = model is not None
        del model
        payload: dict[str, Any] = {
            "ok": True,
            "duration_sec": round(time.perf_counter() - started_at, 3),
            "checks": checks,
        }
    except Exception as exc:
        payload = {
            "ok": False,
            "duration_sec": round(time.perf_counter() - started_at, 3),
            "checks": checks,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


class DesktopApp:
    BACKGROUND = "#F2F5FA"
    CARD = "#FFFFFF"
    NAVY = "#172033"
    MUTED = "#64748B"
    BLUE = "#2563EB"
    BLUE_DARK = "#1D4ED8"
    GREEN = "#059669"
    BORDER = "#D9E2EF"

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(f"{APP_NAME} {APP_VERSION}")
        window_width = min(1100, max(960, self.root.winfo_screenwidth() - 80))
        window_height = min(800, max(700, self.root.winfo_screenheight() - 80))
        position_x = max(0, (self.root.winfo_screenwidth() - window_width) // 2)
        position_y = max(0, (self.root.winfo_screenheight() - window_height) // 2)
        self.root.geometry(
            f"{window_width}x{window_height}+{position_x}+{position_y}"
        )
        self.root.minsize(960, 700)
        self.root.configure(background=self.BACKGROUND)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.events: queue.Queue[tuple[Any, ...]] = queue.Queue()
        self.running = False
        self.worker: threading.Thread | None = None
        self.screenshot_by_item: dict[str, Path] = {}
        self.last_run_dir: Path | None = None
        self.last_report: Path | None = None
        self.last_transcript_markdown: Path | None = None

        self.video_var = tk.StringVar()
        self.output_var = tk.StringVar(value=str(default_output_root()))
        self.video_id_var = tk.StringVar()
        self.preset_var = tk.StringVar(value="智能精准（推荐）")
        self.include_llm_var = tk.BooleanVar(value=True)

        self.result_var = tk.StringVar()
        self.transcript_var = tk.StringVar()

        self.asr_engine_var = tk.StringVar(value=LOCAL_ASR_LABEL)
        self.asr_model_var = tk.StringVar(value="small")
        self.asr_api_key_var = tk.StringVar()
        self.asr_concurrency_var = tk.StringVar(value="3")
        self.asr_upload_consent_var = tk.BooleanVar(value=False)
        self.hotwords_var = tk.StringVar(
            value=(
                "刚体 转动惯量 转动定律 角动量 角动量守恒 "
                "动能定理 质点"
            )
        )
        self.base_url_var = tk.StringVar(value="https://api.openai.com/v1")
        self.llm_model_var = tk.StringVar(value="gpt-4o-mini")
        self.api_key_var = tk.StringVar()
        self.concurrency_var = tk.StringVar(value="5")
        self.llm_include_evidence_var = tk.BooleanVar(value=False)
        self.upload_consent_var = tk.BooleanVar(value=False)

        self.status_var = tk.StringVar(value="请选择一个课堂视频")
        self.progress_var = tk.DoubleVar(value=0)
        self.stage_var = tk.StringVar(value="等待开始")
        self.score_var = tk.StringVar(value="—")
        self.page_count_var = tk.StringVar(value="—")
        self.speech_count_var = tk.StringVar(value="—")

        self._configure_styles()
        self._load_defaults()
        self._build_interface()
        self.root.after(100, self._poll_events)

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure(".", font=("Microsoft YaHei UI", 10))
        style.configure("App.TFrame", background=self.BACKGROUND)
        style.configure("Card.TFrame", background=self.CARD)
        style.configure(
            "Title.TLabel",
            background=self.BACKGROUND,
            foreground=self.NAVY,
            font=("Microsoft YaHei UI", 23, "bold"),
        )
        style.configure(
            "Subtitle.TLabel",
            background=self.BACKGROUND,
            foreground=self.MUTED,
        )
        style.configure(
            "CardTitle.TLabel",
            background=self.CARD,
            foreground=self.NAVY,
            font=("Microsoft YaHei UI", 12, "bold"),
        )
        style.configure(
            "CardText.TLabel",
            background=self.CARD,
            foreground=self.NAVY,
        )
        style.configure(
            "Help.TLabel",
            background=self.CARD,
            foreground=self.MUTED,
            font=("Microsoft YaHei UI", 9),
        )
        style.configure(
            "Metric.TLabel",
            background=self.CARD,
            foreground=self.BLUE_DARK,
            font=("Microsoft YaHei UI", 18, "bold"),
        )
        style.configure(
            "Primary.TButton",
            padding=(18, 11),
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        style.configure("Secondary.TButton", padding=(12, 8))
        style.configure("Treeview", rowheight=32)
        style.configure(
            "Treeview.Heading",
            font=("Microsoft YaHei UI", 9, "bold"),
        )
        style.configure(
            "TNotebook",
            background=self.BACKGROUND,
            borderwidth=0,
        )
        style.configure(
            "TNotebook.Tab",
            padding=(18, 9),
            font=("Microsoft YaHei UI", 10),
        )

    def _build_interface(self) -> None:
        main = ttk.Frame(self.root, style="App.TFrame", padding=(24, 18))
        main.pack(fill="both", expand=True)

        header = ttk.Frame(main, style="App.TFrame")
        header.pack(fill="x")
        title_group = ttk.Frame(header, style="App.TFrame")
        title_group.pack(side="left")
        ttk.Label(title_group, text=APP_NAME, style="Title.TLabel").pack(
            anchor="w"
        )
        ttk.Label(
            title_group,
            text="视频 → PPT页面 → 逐页讲话文字 → 图文关联度评分",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(3, 0))
        self.open_output_button = ttk.Button(
            header,
            text="打开结果目录",
            style="Secondary.TButton",
            command=self._open_output,
            state="disabled",
        )
        self.open_output_button.pack(side="right", pady=(8, 0))
        self.open_report_button = ttk.Button(
            header,
            text="打开评分报告",
            style="Secondary.TButton",
            command=self._open_report,
            state="disabled",
        )
        self.open_report_button.pack(side="right", padx=(0, 8), pady=(8, 0))

        self.notebook = ttk.Notebook(main)
        self.notebook.pack(fill="both", expand=True, pady=(16, 0))
        workflow_tab = ttk.Frame(
            self.notebook, style="App.TFrame", padding=(0, 14)
        )
        continue_tab = ttk.Frame(
            self.notebook, style="App.TFrame", padding=(0, 14)
        )
        settings_tab = ttk.Frame(
            self.notebook, style="App.TFrame", padding=(0, 14)
        )
        self.notebook.add(workflow_tab, text="一键处理")
        self.notebook.add(continue_tab, text="继续已有结果")
        self.notebook.add(settings_tab, text="模型与设置")

        self._build_workflow_tab(workflow_tab)
        self._build_continue_tab(continue_tab)
        self._build_settings_tab(settings_tab)
        self.notebook.select(workflow_tab)

    def _build_workflow_tab(self, parent: ttk.Frame) -> None:
        input_card = ttk.Frame(parent, style="Card.TFrame", padding=18)
        input_card.pack(fill="x")
        input_card.columnconfigure(1, weight=1)
        input_card.columnconfigure(2, minsize=250)
        ttk.Label(
            input_card,
            text="输入与输出",
            style="CardTitle.TLabel",
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))
        self._path_row(
            input_card,
            1,
            "课堂视频",
            self.video_var,
            self._choose_video,
            "选择视频",
        )
        self._path_row(
            input_card,
            2,
            "结果保存到",
            self.output_var,
            self._choose_output,
            "选择目录",
        )
        ttk.Label(
            input_card, text="任务名称", style="CardText.TLabel"
        ).grid(row=3, column=0, sticky="w", padx=(0, 12), pady=5)
        ttk.Entry(input_card, textvariable=self.video_id_var).grid(
            row=3, column=1, sticky="ew", pady=5
        )
        ttk.Label(
            input_card,
            text="默认使用视频文件名，用于创建独立结果文件夹",
            style="Help.TLabel",
            wraplength=235,
            justify="left",
        ).grid(row=3, column=2, sticky="w", padx=(12, 0), pady=5)
        ttk.Label(
            input_card, text="处理模式", style="CardText.TLabel"
        ).grid(row=4, column=0, sticky="w", padx=(0, 12), pady=5)
        ttk.Combobox(
            input_card,
            textvariable=self.preset_var,
            values=("智能精准（推荐）", "快速预览"),
            state="readonly",
            width=22,
        ).grid(row=4, column=1, sticky="w", pady=5)
        ttk.Label(
            input_card,
            text="智能精准适合正式输出；快速预览减少采样，速度更快",
            style="Help.TLabel",
            wraplength=235,
            justify="left",
        ).grid(row=4, column=2, sticky="w", padx=(12, 0), pady=5)

        llm_row = ttk.Frame(input_card, style="Card.TFrame")
        llm_row.grid(
            row=5, column=0, columnspan=3, sticky="ew", pady=(8, 0)
        )
        ttk.Checkbutton(
            llm_row,
            text="完成文字转写后继续进行 LLM 图文关联度评分",
            variable=self.include_llm_var,
        ).pack(side="left")
        ttk.Label(
            llm_row,
            text="需要先在“模型与设置”中填写服务、模型和 API Key",
            style="Help.TLabel",
        ).pack(side="left", padx=(12, 0))

        actions = ttk.Frame(input_card, style="Card.TFrame")
        actions.grid(
            row=6, column=0, columnspan=3, sticky="ew", pady=(14, 0)
        )
        self.full_button = ttk.Button(
            actions,
            text="开始完整处理",
            style="Primary.TButton",
            command=self._start_full,
        )
        self.full_button.pack(side="left")
        self.detect_button = ttk.Button(
            actions,
            text="仅提取PPT页面",
            style="Secondary.TButton",
            command=self._start_detect_only,
        )
        self.detect_button.pack(side="left", padx=(10, 0))

        metrics = ttk.Frame(parent, style="Card.TFrame", padding=16)
        metrics.pack(fill="x", pady=(14, 0))
        for column in range(4):
            metrics.columnconfigure(column, weight=1)
        self._metric(metrics, 0, "当前阶段", self.stage_var)
        self._metric(metrics, 1, "PPT页数", self.page_count_var)
        self._metric(metrics, 2, "讲话段数", self.speech_count_var)
        self._metric(metrics, 3, "关联度总分", self.score_var)

        status_card = ttk.Frame(parent, style="Card.TFrame", padding=16)
        status_card.pack(fill="x", pady=(14, 0))
        status_header = ttk.Frame(status_card, style="Card.TFrame")
        status_header.pack(fill="x")
        ttk.Label(
            status_header, text="处理进度", style="CardTitle.TLabel"
        ).pack(side="left")
        ttk.Label(
            status_header, textvariable=self.status_var, style="Help.TLabel"
        ).pack(side="right")
        ttk.Progressbar(
            status_card, variable=self.progress_var, maximum=100
        ).pack(fill="x", pady=(12, 9))
        self.log = ScrolledText(
            status_card,
            height=3,
            wrap="word",
            relief="flat",
            background="#F8FAFC",
            foreground=self.NAVY,
            font=("Microsoft YaHei UI", 9),
            padx=10,
            pady=8,
            state="disabled",
        )
        self.log.pack(fill="x")

        result_card = ttk.Frame(parent, style="Card.TFrame", padding=16)
        result_card.pack(fill="both", expand=True, pady=(14, 0))
        ttk.Label(
            result_card, text="逐页结果", style="CardTitle.TLabel"
        ).pack(anchor="w", pady=(0, 8))
        columns = ("page", "time", "speech", "confidence", "score", "level")
        self.result_tree = ttk.Treeview(
            result_card,
            columns=columns,
            show="headings",
            height=5,
        )
        headings = {
            "page": "页码",
            "time": "PPT时间",
            "speech": "讲话段数",
            "confidence": "页面置信度",
            "score": "关联分数",
            "level": "评分等级",
        }
        widths = {
            "page": 65,
            "time": 230,
            "speech": 95,
            "confidence": 120,
            "score": 100,
            "level": 150,
        }
        for column in columns:
            self.result_tree.heading(column, text=headings[column])
            self.result_tree.column(
                column,
                width=widths[column],
                anchor="center",
                minwidth=60,
            )
        scrollbar = ttk.Scrollbar(
            result_card, orient="vertical", command=self.result_tree.yview
        )
        self.result_tree.configure(yscrollcommand=scrollbar.set)
        self.result_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.result_tree.bind("<Double-1>", self._open_selected_screenshot)

    def _build_continue_tab(self, parent: ttk.Frame) -> None:
        card = ttk.Frame(parent, style="Card.TFrame", padding=20)
        card.pack(fill="x")
        card.columnconfigure(1, weight=1)
        ttk.Label(
            card, text="继续处理中间结果", style="CardTitle.TLabel"
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 6))
        ttk.Label(
            card,
            text=(
                "页面已经提取时可直接转写；文字已经生成时可直接评分，"
                "无需重新处理前面的步骤。"
            ),
            style="Help.TLabel",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 14))
        self._path_row(
            card,
            2,
            "课堂视频",
            self.video_var,
            self._choose_video,
            "选择视频",
        )
        self._path_row(
            card,
            3,
            "页面结果",
            self.result_var,
            self._choose_result,
            "选择 result.json",
        )
        self._path_row(
            card,
            4,
            "转写结果",
            self.transcript_var,
            self._choose_transcript,
            "选择 transcript.json",
        )
        actions = ttk.Frame(card, style="Card.TFrame")
        actions.grid(
            row=5, column=0, columnspan=3, sticky="ew", pady=(16, 0)
        )
        self.transcribe_button = ttk.Button(
            actions,
            text="从页面结果开始转写",
            style="Primary.TButton",
            command=self._start_transcribe_only,
        )
        self.transcribe_button.pack(side="left")
        self.evaluate_button = ttk.Button(
            actions,
            text="从转写结果开始评分",
            style="Secondary.TButton",
            command=self._start_evaluate_only,
        )
        self.evaluate_button.pack(side="left", padx=(10, 0))
        ttk.Label(
            card,
            text="运行状态和逐页结果会显示在“一键处理”页签。",
            style="Help.TLabel",
        ).grid(row=6, column=0, columnspan=3, sticky="w", pady=(14, 0))

    def _build_settings_tab(self, parent: ttk.Frame) -> None:
        asr_card = ttk.Frame(parent, style="Card.TFrame", padding=18)
        asr_card.pack(fill="x")
        asr_card.columnconfigure(1, weight=1)
        ttk.Label(
            asr_card, text="语音识别", style="CardTitle.TLabel"
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))
        ttk.Label(
            asr_card, text="识别方式", style="CardText.TLabel"
        ).grid(row=1, column=0, sticky="w", padx=(0, 12), pady=5)
        ttk.Combobox(
            asr_card,
            textvariable=self.asr_engine_var,
            values=(LOCAL_ASR_LABEL, MIMO_ASR_LABEL),
            state="readonly",
            width=28,
        ).grid(row=1, column=1, sticky="w", pady=5)
        ttk.Label(
            asr_card,
            text=(
                "完整处理时，每确认一页就立即启动云端转写，"
                "转写完成后立即评分；本地模式不上传课堂音频"
            ),
            style="Help.TLabel",
            wraplength=270,
            justify="left",
        ).grid(row=1, column=2, sticky="w", padx=(12, 0), pady=5)
        ttk.Label(
            asr_card, text="本地模型", style="CardText.TLabel"
        ).grid(row=2, column=0, sticky="w", padx=(0, 12), pady=5)
        ttk.Combobox(
            asr_card,
            textvariable=self.asr_model_var,
            values=("tiny", "base", "small", "medium", "large-v3"),
            state="readonly",
            width=20,
        ).grid(row=2, column=1, sticky="w", pady=5)
        ttk.Label(
            asr_card,
            text="仅本地模式使用；发布版内置 small 模型",
            style="Help.TLabel",
            wraplength=270,
            justify="left",
        ).grid(row=2, column=2, sticky="w", padx=(12, 0), pady=5)
        ttk.Label(
            asr_card, text="专业词汇", style="CardText.TLabel"
        ).grid(row=3, column=0, sticky="w", padx=(0, 12), pady=5)
        ttk.Entry(asr_card, textvariable=self.hotwords_var).grid(
            row=3, column=1, sticky="ew", pady=5
        )
        ttk.Label(
            asr_card,
            text="本地模式使用；用空格分隔",
            style="Help.TLabel",
            wraplength=270,
            justify="left",
        ).grid(row=3, column=2, sticky="w", padx=(12, 0), pady=5)
        ttk.Label(
            asr_card, text="小米ASR Key", style="CardText.TLabel"
        ).grid(row=4, column=0, sticky="w", padx=(0, 12), pady=5)
        ttk.Entry(
            asr_card,
            textvariable=self.asr_api_key_var,
            show="●",
        ).grid(row=4, column=1, sticky="ew", pady=5)
        ttk.Label(
            asr_card,
            text="只在内存中使用；留空时读取MIMO_API_KEY或复用下方Key",
            style="Help.TLabel",
            wraplength=270,
            justify="left",
        ).grid(row=4, column=2, sticky="w", padx=(12, 0), pady=5)
        ttk.Label(
            asr_card, text="云端并发", style="CardText.TLabel"
        ).grid(row=5, column=0, sticky="w", padx=(0, 12), pady=5)
        ttk.Spinbox(
            asr_card,
            from_=1,
            to=10,
            textvariable=self.asr_concurrency_var,
            width=8,
        ).grid(row=5, column=1, sticky="w", pady=5)
        ttk.Label(
            asr_card,
            text="默认3；服务限流时调低",
            style="Help.TLabel",
            wraplength=270,
            justify="left",
        ).grid(row=5, column=2, sticky="w", padx=(12, 0), pady=5)
        ttk.Checkbutton(
            asr_card,
            text=(
                "我确认允许把按PPT时间截取的临时音频发送给小米MiMo；"
                "请求结束后立即删除音频"
            ),
            variable=self.asr_upload_consent_var,
        ).grid(row=6, column=0, columnspan=3, sticky="w", pady=(10, 0))

        llm_card = ttk.Frame(parent, style="Card.TFrame", padding=18)
        llm_card.pack(fill="x", pady=(14, 0))
        llm_card.columnconfigure(1, weight=1)
        ttk.Label(
            llm_card, text="多模态 LLM 服务", style="CardTitle.TLabel"
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))
        self._entry_row(
            llm_card,
            1,
            "Base URL",
            self.base_url_var,
            "OpenAI兼容地址，可填到 /v1 或 /chat/completions",
        )
        self._entry_row(
            llm_card,
            2,
            "模型名称",
            self.llm_model_var,
            "必须支持图片输入",
        )
        ttk.Label(
            llm_card, text="API Key", style="CardText.TLabel"
        ).grid(row=3, column=0, sticky="w", padx=(0, 12), pady=5)
        ttk.Entry(
            llm_card, textvariable=self.api_key_var, show="●"
        ).grid(row=3, column=1, sticky="ew", pady=5)
        ttk.Label(
            llm_card,
            text="只保存在当前应用内存，不写入设置或结果",
            style="Help.TLabel",
            wraplength=270,
            justify="left",
        ).grid(row=3, column=2, sticky="w", padx=(12, 0), pady=5)
        ttk.Label(
            llm_card, text="并发数量", style="CardText.TLabel"
        ).grid(row=4, column=0, sticky="w", padx=(0, 12), pady=5)
        ttk.Spinbox(
            llm_card,
            from_=1,
            to=10,
            textvariable=self.concurrency_var,
            width=8,
        ).grid(row=4, column=1, sticky="w", pady=5)
        ttk.Label(
            llm_card,
            text="默认5；服务限流时调低",
            style="Help.TLabel",
            wraplength=270,
            justify="left",
        ).grid(row=4, column=2, sticky="w", padx=(12, 0), pady=5)
        ttk.Checkbutton(
            llm_card,
            text="返回详细对应证据（增加Token消耗）",
            variable=self.llm_include_evidence_var,
        ).grid(row=5, column=0, columnspan=3, sticky="w", pady=(10, 0))
        ttk.Checkbutton(
            llm_card,
            text=(
                "我确认允许把PPT截图和对应课堂转写发送给上述LLM服务"
            ),
            variable=self.upload_consent_var,
        ).grid(row=6, column=0, columnspan=3, sticky="w", pady=(8, 0))

        footer = ttk.Frame(parent, style="App.TFrame")
        footer.pack(fill="x", pady=(14, 0))
        ttk.Button(
            footer,
            text="保存非敏感设置",
            style="Secondary.TButton",
            command=self._save_settings,
        ).pack(side="left")
        ttk.Label(
            footer,
            text="API Key和上传授权不会保存；每次启动需要重新确认。",
            style="Subtitle.TLabel",
        ).pack(side="left", padx=(12, 0))

    @staticmethod
    def _metric(
        parent: ttk.Frame,
        column: int,
        label: str,
        variable: tk.StringVar,
    ) -> None:
        cell = ttk.Frame(parent, style="Card.TFrame")
        cell.grid(column=column, row=0, sticky="ew", padx=(4, 18))
        ttk.Label(cell, text=label, style="Help.TLabel").pack(anchor="w")
        ttk.Label(cell, textvariable=variable, style="Metric.TLabel").pack(
            anchor="w", pady=(3, 0)
        )

    @staticmethod
    def _path_row(
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.StringVar,
        command: Any,
        button_text: str,
    ) -> None:
        ttk.Label(parent, text=label, style="CardText.TLabel").grid(
            row=row, column=0, sticky="w", padx=(0, 12), pady=5
        )
        ttk.Entry(parent, textvariable=variable).grid(
            row=row, column=1, sticky="ew", pady=5
        )
        ttk.Button(parent, text=button_text, command=command).grid(
            row=row, column=2, sticky="e", padx=(12, 0), pady=5
        )

    @staticmethod
    def _entry_row(
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.StringVar,
        help_text: str,
    ) -> None:
        ttk.Label(parent, text=label, style="CardText.TLabel").grid(
            row=row, column=0, sticky="w", padx=(0, 12), pady=5
        )
        ttk.Entry(parent, textvariable=variable).grid(
            row=row, column=1, sticky="ew", pady=5
        )
        ttk.Label(
            parent,
            text=help_text,
            style="Help.TLabel",
            wraplength=270,
            justify="left",
        ).grid(
            row=row, column=2, sticky="w", padx=(12, 0), pady=5
        )

    def _load_defaults(self) -> None:
        llm_path = resource_path("config/llm_evaluation.json")
        if llm_path.is_file():
            try:
                llm_config = LLMEvaluationConfig.from_file(llm_path)
                self.base_url_var.set(llm_config.base_url)
                self.llm_model_var.set(llm_config.model)
                self.concurrency_var.set(str(llm_config.max_concurrency))
                self.llm_include_evidence_var.set(
                    llm_config.include_evidence
                )
            except (OSError, ValueError):
                pass
        saved_path = settings_path()
        if not saved_path.is_file():
            return
        try:
            data = json.loads(saved_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(data, dict):
            return
        mapping = {
            "output_root": self.output_var,
            "preset": self.preset_var,
            "asr_engine": self.asr_engine_var,
            "asr_model": self.asr_model_var,
            "asr_concurrency": self.asr_concurrency_var,
            "hotwords": self.hotwords_var,
            "base_url": self.base_url_var,
            "llm_model": self.llm_model_var,
            "concurrency": self.concurrency_var,
        }
        for key, variable in mapping.items():
            value = data.get(key)
            if isinstance(value, str) and value:
                variable.set(value)
        include_llm = data.get("include_llm")
        if isinstance(include_llm, bool):
            self.include_llm_var.set(include_llm)
        include_evidence = data.get("llm_include_evidence")
        if isinstance(include_evidence, bool):
            self.llm_include_evidence_var.set(include_evidence)

    def _save_settings(self) -> None:
        payload = {
            "output_root": self.output_var.get().strip(),
            "preset": self.preset_var.get(),
            "include_llm": bool(self.include_llm_var.get()),
            "asr_engine": self.asr_engine_var.get(),
            "asr_model": self.asr_model_var.get(),
            "asr_concurrency": self.asr_concurrency_var.get().strip(),
            "hotwords": self.hotwords_var.get().strip(),
            "base_url": self.base_url_var.get().strip(),
            "llm_model": self.llm_model_var.get().strip(),
            "concurrency": self.concurrency_var.get().strip(),
            "llm_include_evidence": bool(
                self.llm_include_evidence_var.get()
            ),
        }
        path = settings_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            messagebox.showerror("保存失败", str(exc), parent=self.root)
            return
        messagebox.showinfo(
            "设置已保存",
            "非敏感设置已保存。API Key和上传授权未保存。",
            parent=self.root,
        )

    def _choose_video(self) -> None:
        selected = filedialog.askopenfilename(
            title="选择课堂视频",
            filetypes=VIDEO_FILE_TYPES,
            parent=self.root,
        )
        if not selected:
            return
        self.video_var.set(selected)
        self.video_id_var.set(Path(selected).stem)
        paths = build_workflow_paths(
            self.output_var.get().strip() or default_output_root(),
            Path(selected).stem,
        )
        self.result_var.set(str(paths.result_json))
        self.transcript_var.set(str(paths.transcript_json))

    def _choose_output(self) -> None:
        selected = filedialog.askdirectory(
            title="选择结果保存目录",
            parent=self.root,
        )
        if selected:
            self.output_var.set(selected)

    def _choose_result(self) -> None:
        selected = filedialog.askopenfilename(
            title="选择 result.json",
            filetypes=[("页面结果", "result.json"), ("JSON文件", "*.json")],
            parent=self.root,
        )
        if selected:
            self.result_var.set(selected)
            self.transcript_var.set(str(Path(selected).parent / "transcript.json"))

    def _choose_transcript(self) -> None:
        selected = filedialog.askopenfilename(
            title="选择 transcript.json",
            filetypes=[
                ("语音转写结果", "transcript.json"),
                ("JSON文件", "*.json"),
            ],
            parent=self.root,
        )
        if selected:
            self.transcript_var.set(selected)

    def _bundled_binary(self, name: str) -> str | None:
        candidate = resource_path(Path("tools") / name)
        return str(candidate) if candidate.is_file() else None

    def _detector_config(self) -> DetectorConfig:
        return detector_config_for_preset(
            self.preset_var.get(),
            ffmpeg_path=self._bundled_binary("ffmpeg.exe"),
            ffprobe_path=self._bundled_binary("ffprobe.exe"),
        )

    def _transcription_config(self) -> TranscriptionConfig:
        config_path = resource_path("config/transcription.json")
        base = (
            TranscriptionConfig.from_file(config_path)
            if config_path.is_file()
            else TranscriptionConfig()
        )
        model_root = resource_path("models/faster-whisper")
        config = replace(
            base,
            engine=(
                "mimo-cloud"
                if self.asr_engine_var.get() == MIMO_ASR_LABEL
                else "faster-whisper"
            ),
            model=self.asr_model_var.get().strip(),
            hotwords=self.hotwords_var.get().strip() or None,
            mimo_max_concurrency=int(self.asr_concurrency_var.get()),
            model_download_root=str(model_root),
            ffmpeg_path=self._bundled_binary("ffmpeg.exe"),
        )
        config.validate()
        return config

    def _require_asr_api_key(
        self,
        config: TranscriptionConfig,
    ) -> str | None:
        if config.engine != "mimo-cloud":
            return None
        if not self.asr_upload_consent_var.get():
            raise ValueError(
                "请在“模型与设置”中确认允许发送临时课堂音频给小米MiMo。"
            )
        return resolve_mimo_api_key(
            config.mimo_api_key_env,
            self.asr_api_key_var.get().strip()
            or self.api_key_var.get().strip()
            or None,
        )

    def _llm_config(self) -> LLMEvaluationConfig:
        config_path = resource_path("config/llm_evaluation.json")
        base = (
            LLMEvaluationConfig.from_file(config_path)
            if config_path.is_file()
            else LLMEvaluationConfig()
        )
        config = replace(
            base,
            base_url=self.base_url_var.get().strip(),
            model=self.llm_model_var.get().strip(),
            max_concurrency=int(self.concurrency_var.get()),
            include_evidence=bool(self.llm_include_evidence_var.get()),
        )
        config.validate()
        return config

    def _require_llm_inputs(self) -> tuple[LLMEvaluationConfig, str]:
        if not self.upload_consent_var.get():
            raise ValueError(
                "请在“模型与设置”中确认允许发送PPT截图和课堂转写。"
            )
        api_key = (
            self.api_key_var.get().strip()
            or os.environ.get("LLM_API_KEY", "").strip()
        )
        if not api_key:
            raise ValueError("请输入API Key，或设置环境变量 LLM_API_KEY。")
        return self._llm_config(), api_key

    def _start_full(self) -> None:
        if self.running:
            return
        try:
            video = self._valid_video()
            output_root = self._valid_output()
            video_id = sanitize_video_id(
                self.video_id_var.get(), video.stem
            )
            detector_config = self._detector_config()
            transcription_config = self._transcription_config()
            asr_api_key = self._require_asr_api_key(
                transcription_config
            )
            llm_inputs = (
                self._require_llm_inputs()
                if self.include_llm_var.get()
                else None
            )
        except (OSError, TypeError, ValueError) as exc:
            messagebox.showerror("无法开始", str(exc), parent=self.root)
            return
        paths = build_workflow_paths(output_root, video_id)
        self.result_var.set(str(paths.result_json))
        self.transcript_var.set(str(paths.transcript_json))
        self._begin(
            "full",
            video=video,
            paths=paths,
            detector_config=detector_config,
            transcription_config=transcription_config,
            asr_api_key=asr_api_key,
            llm_inputs=llm_inputs,
        )

    def _start_detect_only(self) -> None:
        if self.running:
            return
        try:
            video = self._valid_video()
            output_root = self._valid_output()
            video_id = sanitize_video_id(
                self.video_id_var.get(), video.stem
            )
            detector_config = self._detector_config()
        except (OSError, TypeError, ValueError) as exc:
            messagebox.showerror("无法开始", str(exc), parent=self.root)
            return
        paths = build_workflow_paths(output_root, video_id)
        self.result_var.set(str(paths.result_json))
        self._begin(
            "detect",
            video=video,
            paths=paths,
            detector_config=detector_config,
        )

    def _start_transcribe_only(self) -> None:
        if self.running:
            return
        try:
            video = self._valid_video()
            result_path = Path(self.result_var.get().strip())
            if not result_path.is_file():
                raise ValueError("请选择有效的 result.json。")
            transcription_config = self._transcription_config()
            asr_api_key = self._require_asr_api_key(
                transcription_config
            )
            paths = build_workflow_paths(
                result_path.parent.parent,
                result_path.parent.name,
            )
        except (OSError, TypeError, ValueError) as exc:
            messagebox.showerror("无法开始", str(exc), parent=self.root)
            return
        self.transcript_var.set(str(paths.transcript_json))
        self._begin(
            "transcribe",
            video=video,
            paths=paths,
            result_path=result_path,
            transcription_config=transcription_config,
            asr_api_key=asr_api_key,
        )

    def _start_evaluate_only(self) -> None:
        if self.running:
            return
        try:
            transcript_path = Path(self.transcript_var.get().strip())
            if not transcript_path.is_file():
                raise ValueError("请选择有效的 transcript.json。")
            llm_inputs = self._require_llm_inputs()
            paths = build_workflow_paths(
                transcript_path.parent.parent,
                transcript_path.parent.name,
            )
        except (OSError, TypeError, ValueError) as exc:
            messagebox.showerror("无法开始", str(exc), parent=self.root)
            return
        self._begin(
            "evaluate",
            paths=paths,
            transcript_path=transcript_path,
            llm_inputs=llm_inputs,
        )

    def _valid_video(self) -> Path:
        video = Path(self.video_var.get().strip())
        if not video.is_file():
            raise ValueError("请选择一个有效的课堂视频。")
        return video

    def _valid_output(self) -> Path:
        text = self.output_var.get().strip()
        if not text:
            raise ValueError("请选择结果保存目录。")
        output = Path(text)
        output.mkdir(parents=True, exist_ok=True)
        return output

    def _begin(self, mode: str, **kwargs: Any) -> None:
        self.running = True
        self.progress_var.set(0)
        self.stage_var.set("准备")
        self.status_var.set("正在准备处理")
        self._set_buttons_state("disabled")
        self.open_output_button.configure(state="disabled")
        self.open_report_button.configure(state="disabled")
        self._clear_log()
        self._clear_results()
        self._append_log(f"任务模式：{mode}")
        paths = kwargs.get("paths")
        if isinstance(paths, WorkflowPaths):
            self._append_log(f"结果目录：{paths.run_dir}")
        self.notebook.select(0)
        self.worker = threading.Thread(
            target=self._run_worker,
            args=(mode, kwargs),
            daemon=True,
        )
        self.worker.start()

    def _run_worker(self, mode: str, kwargs: dict[str, Any]) -> None:
        artifacts: dict[str, Any] = {}
        cloud_pipeline: CloudPagePipeline | None = None
        try:
            paths: WorkflowPaths = kwargs["paths"]
            video: Path | None = kwargs.get("video")
            transcription_config: TranscriptionConfig | None = kwargs.get(
                "transcription_config"
            )
            streaming_cloud = (
                mode == "full"
                and transcription_config is not None
                and transcription_config.engine == "mimo-cloud"
            )
            if streaming_cloud:
                if video is None:
                    raise ValueError("完整云端流水线缺少视频文件。")
                llm_config: LLMEvaluationConfig | None = None
                llm_api_key: str | None = None
                if kwargs.get("llm_inputs") is not None:
                    llm_config, llm_api_key = kwargs["llm_inputs"]

                def pipeline_progress(
                    stage: str,
                    message: str,
                    completed: int,
                    total: int,
                ) -> None:
                    ratio = completed / max(total, 1)
                    if stage == "LLM关联度评分":
                        progress = 55 + ratio * 40
                    else:
                        progress = 30 + ratio * (
                            45 if llm_config is not None else 65
                        )
                    self._put_progress(stage, message, progress)

                cloud_pipeline = CloudPagePipeline(
                    video_path=video,
                    result_path=paths.result_json,
                    output_dir=paths.run_dir,
                    transcription_config=transcription_config,
                    asr_api_key=kwargs.get("asr_api_key") or "",
                    llm_config=llm_config,
                    llm_api_key=llm_api_key,
                    progress_callback=pipeline_progress,
                )
            if mode in {"full", "detect"}:
                self._put_progress("PPT页面识别", "正在分析视频页面", 1)
                detector = VideoPageDetector(kwargs["detector_config"])
                detection = detector.run(
                    video,
                    output_root=paths.output_root,
                    video_id=paths.video_id,
                    progress_callback=lambda message, value: self._put_progress(
                        "PPT页面识别",
                        message,
                        (
                            float(value or 0)
                            * (
                                60
                                if mode == "full" and streaming_cloud
                                else (30 if mode == "full" else 100)
                            )
                        ),
                    ),
                    page_ready_callback=(
                        cloud_pipeline.submit_page
                        if cloud_pipeline is not None
                        else None
                    ),
                )
                artifacts["detection"] = detection
            elif mode == "transcribe":
                artifacts["detection"] = self._read_json(
                    Path(kwargs["result_path"])
                )
            if cloud_pipeline is not None:
                self._put_progress(
                    "云端流水线",
                    "PPT检测完成，正在等待剩余页面转写与评分",
                    60,
                )
                transcript, evaluation = cloud_pipeline.finish(detection)
                artifacts["transcript"] = transcript
                if evaluation is not None:
                    artifacts["evaluation"] = evaluation
                cloud_pipeline = None
            elif mode in {"full", "transcribe"}:
                result_path = (
                    paths.result_json
                    if mode == "full"
                    else Path(kwargs["result_path"])
                )
                start = 30 if mode == "full" else 0
                span = 45 if mode == "full" else 100
                transcription = transcribe_video_pages(
                    video,
                    result_path,
                    config=kwargs["transcription_config"],
                    output_dir=paths.run_dir,
                    api_key=kwargs.get("asr_api_key"),
                    progress_callback=lambda message, value: self._put_progress(
                        "语音转写",
                        message,
                        start + float(value or 0) * span,
                    ),
                )
                artifacts["transcript"] = transcription
            if mode == "evaluate":
                transcript_path = Path(kwargs["transcript_path"])
                artifacts["transcript"] = self._read_json(transcript_path)
            if (
                mode == "full"
                and not streaming_cloud
                and kwargs.get("llm_inputs") is not None
            ):
                llm_config, api_key = kwargs["llm_inputs"]
                evaluation = evaluate_transcript(
                    paths.transcript_json,
                    config=llm_config,
                    output_dir=paths.evaluation_dir,
                    api_key=api_key,
                    progress_callback=lambda message, completed, total: (
                        self._put_progress(
                            "LLM关联度评分",
                            f"{completed}/{total}：{message}",
                            75 + completed / max(total, 1) * 25,
                        )
                    ),
                )
                artifacts["evaluation"] = evaluation
            elif mode == "evaluate":
                llm_config, api_key = kwargs["llm_inputs"]
                evaluation = evaluate_transcript(
                    Path(kwargs["transcript_path"]),
                    config=llm_config,
                    output_dir=paths.evaluation_dir,
                    api_key=api_key,
                    progress_callback=lambda message, completed, total: (
                        self._put_progress(
                            "LLM关联度评分",
                            f"{completed}/{total}：{message}",
                            completed / max(total, 1) * 100,
                        )
                    ),
                )
                artifacts["evaluation"] = evaluation
            self.events.put(("complete", mode, paths, artifacts))
        except Exception as exc:
            if cloud_pipeline is not None:
                cloud_pipeline.abort()
            self.events.put(
                ("error", str(exc), traceback.format_exc())
            )
        finally:
            kwargs["llm_inputs"] = None
            kwargs["asr_api_key"] = None

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"JSON根节点必须是对象：{path}")
        return data

    def _put_progress(
        self,
        stage: str,
        message: str,
        progress: float,
    ) -> None:
        self.events.put(
            (
                "progress",
                stage,
                message,
                max(0.0, min(100.0, progress)),
            )
        )

    def _poll_events(self) -> None:
        try:
            while True:
                event = self.events.get_nowait()
                kind = event[0]
                if kind == "progress":
                    _, stage, message, progress = event
                    self.stage_var.set(str(stage))
                    self.status_var.set(str(message))
                    self.progress_var.set(
                        max(
                            float(self.progress_var.get()),
                            float(progress),
                        )
                    )
                    self._append_log(str(message))
                elif kind == "complete":
                    _, mode, paths, artifacts = event
                    self._handle_complete(
                        str(mode),
                        paths,
                        dict(artifacts),
                    )
                elif kind == "error":
                    _, message, details = event
                    self._handle_error(str(message), str(details))
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def _handle_complete(
        self,
        mode: str,
        paths: WorkflowPaths,
        artifacts: dict[str, Any],
    ) -> None:
        self.running = False
        self._set_buttons_state("normal")
        self.progress_var.set(100)
        self.stage_var.set("完成")
        self.status_var.set("处理完成")
        self.last_run_dir = paths.run_dir
        self.open_output_button.configure(state="normal")

        detection = artifacts.get("detection")
        transcript = artifacts.get("transcript")
        evaluation = artifacts.get("evaluation")
        self._populate_rows(
            combine_page_rows(detection, transcript, evaluation)
        )

        if detection:
            page_count = len(detection.get("pages", []))
            self.page_count_var.set(str(page_count))
            self.result_var.set(str(paths.result_json))
        if transcript:
            count = transcript.get("transcription", {}).get(
                "utterance_count", 0
            )
            self.speech_count_var.set(str(count))
            self.transcript_var.set(str(paths.transcript_json))
            artifact_paths = transcript.get("artifacts", {})
            markdown = artifact_paths.get("page_transcript_markdown")
            if markdown:
                self.last_transcript_markdown = Path(str(markdown))
        if evaluation:
            summary = evaluation.get("summary", {})
            score = summary.get("strict_overall_score")
            self.score_var.set("—" if score is None else str(score))
            report = evaluation.get("artifacts", {}).get("report_markdown")
            if report:
                self.last_report = Path(str(report))
                self.open_report_button.configure(state="normal")
        self.api_key_var.set("")
        self.asr_api_key_var.set("")
        self.asr_upload_consent_var.set(False)
        self.upload_consent_var.set(False)
        self._append_log("处理完成。API Key已从界面内存清空。")
        messagebox.showinfo(
            "处理完成",
            self._completion_message(mode, artifacts, paths),
            parent=self.root,
        )

    @staticmethod
    def _completion_message(
        mode: str,
        artifacts: Mapping[str, Any],
        paths: WorkflowPaths,
    ) -> str:
        lines = ["任务已经完成。"]
        detection = artifacts.get("detection")
        transcript = artifacts.get("transcript")
        evaluation = artifacts.get("evaluation")
        if isinstance(detection, Mapping):
            lines.append(f"PPT页数：{len(detection.get('pages', []))}")
        if isinstance(transcript, Mapping):
            lines.append(
                "讲话段数："
                + str(
                    transcript.get("transcription", {}).get(
                        "utterance_count", 0
                    )
                )
            )
        if isinstance(evaluation, Mapping):
            summary = evaluation.get("summary", {})
            lines.append(
                f"关联度总分：{summary.get('strict_overall_score', '—')}"
            )
            lines.append(
                f"评分成功页：{summary.get('scored_pages', 0)}/"
                f"{summary.get('total_pages', 0)}"
            )
        lines.extend(["", f"结果目录：{paths.run_dir}"])
        return "\n".join(lines)

    def _handle_error(self, message: str, details: str) -> None:
        self.running = False
        self._set_buttons_state("normal")
        self.stage_var.set("失败")
        self.status_var.set("处理失败")
        self.api_key_var.set("")
        self.asr_api_key_var.set("")
        self.asr_upload_consent_var.set(False)
        self.upload_consent_var.set(False)
        self._append_log(f"处理失败：{message}")
        self._append_log(details)
        messagebox.showerror(
            "处理失败",
            f"{message}\n\n详细信息已显示在处理日志中。",
            parent=self.root,
        )

    def _populate_rows(self, rows: list[dict[str, Any]]) -> None:
        self._clear_results()
        for index, row in enumerate(rows):
            start = float(row.get("start_sec", 0))
            end = float(row.get("end_sec", 0))
            item = self.result_tree.insert(
                "",
                "end",
                values=(
                    row["page_id"],
                    f"{format_timestamp(start)} ～ {format_timestamp(end)}",
                    row.get("utterance_count", "—"),
                    row.get("confidence", "—"),
                    row.get("score", "—"),
                    row.get("level", "—"),
                ),
                tags=("even" if index % 2 == 0 else "odd",),
            )
            screenshot = row.get("screenshot_path")
            if screenshot:
                self.screenshot_by_item[item] = Path(str(screenshot))
        self.result_tree.tag_configure("even", background="#F8FAFC")
        self.result_tree.tag_configure("odd", background="#FFFFFF")

    def _clear_results(self) -> None:
        for item in self.result_tree.get_children():
            self.result_tree.delete(item)
        self.screenshot_by_item.clear()

    def _append_log(self, message: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", message.rstrip() + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _set_buttons_state(self, state: str) -> None:
        for button in (
            self.full_button,
            self.detect_button,
            self.transcribe_button,
            self.evaluate_button,
        ):
            button.configure(state=state)

    def _open_output(self) -> None:
        if self.last_run_dir and self.last_run_dir.exists():
            open_external(self.last_run_dir)

    def _open_report(self) -> None:
        if self.last_report and self.last_report.is_file():
            open_external(self.last_report)

    def _open_selected_screenshot(self, _: tk.Event[tk.Misc]) -> None:
        selected = self.result_tree.selection()
        if not selected:
            return
        path = self.screenshot_by_item.get(selected[0])
        if path and path.is_file():
            open_external(path)

    def _on_close(self) -> None:
        if self.running and not messagebox.askyesno(
            "确认退出",
            "任务仍在运行。现在退出会中断处理，确定退出吗？",
            parent=self.root,
        ):
            return
        self.api_key_var.set("")
        self.asr_api_key_var.set("")
        self.root.destroy()


def main() -> int:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"Unable to start GUI: {exc}", file=sys.stderr)
        return 2
    DesktopApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
