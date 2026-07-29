from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Callable, Mapping

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from .config import DetectorConfig
from .pipeline import VideoPageDetector


PARAMETER_FIELDS: tuple[tuple[str, str, str, Callable[[str], object]], ...] = (
    (
        "temporal_sample_interval_sec",
        "时序采样间隔",
        "单位：秒，默认 2",
        float,
    ),
    (
        "temporal_confirmation_sec",
        "换页确认时长",
        "单位：秒，默认 10",
        float,
    ),
    (
        "temporal_changed_block_ratio",
        "时序换页比例",
        "默认 0.50",
        float,
    ),
    (
        "temporal_same_content_similarity",
        "同页内容相似度",
        "默认 0.80；合并缩放、重排等同页动画",
        float,
    ),
    ("min_page_duration_sec", "最短页面停留", "单位：秒", float),
    ("screen_crop_left_ratio", "投影区左边距", "画面比例，默认 0.10", float),
    ("screen_crop_top_ratio", "投影区上边距", "画面比例，默认 0.02", float),
    ("screen_crop_right_ratio", "投影区右边距", "画面比例，默认 0.10", float),
    ("screen_crop_bottom_ratio", "投影区下边距", "画面比例，默认 0.10", float),
)


def format_seconds(seconds: float) -> str:
    total_milliseconds = int(round(float(seconds) * 1000))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs = remainder / 1000
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    return f"{minutes:02d}:{secs:06.3f}"


def config_from_gui_values(
    base: DetectorConfig,
    values: Mapping[str, str],
    *,
    ffmpeg_path: str = "",
    ffprobe_path: str = "",
) -> DetectorConfig:
    converters = {key: converter for key, _, _, converter in PARAMETER_FIELDS}
    overrides: dict[str, object] = {}
    for key, converter in converters.items():
        raw = values.get(key, "").strip()
        if not raw:
            raise ValueError(f"参数“{key}”不能为空")
        try:
            overrides[key] = converter(raw)
        except ValueError as exc:
            raise ValueError(f"参数“{key}”格式不正确：{raw}") from exc
    overrides["ffmpeg_path"] = ffmpeg_path.strip() or None
    overrides["ffprobe_path"] = ffprobe_path.strip() or None
    config = replace(base, **overrides)
    config.validate()
    return config


def open_external(path: Path) -> None:
    target = str(path.resolve())
    if sys.platform == "win32":
        os.startfile(target)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", target])
    else:
        subprocess.Popen(["xdg-open", target])


class DetectorApp:
    BACKGROUND = "#F4F7FB"
    CARD = "#FFFFFF"
    PRIMARY = "#2563EB"
    PRIMARY_DARK = "#1D4ED8"
    TEXT = "#172033"
    MUTED = "#64748B"
    BORDER = "#DCE3ED"
    SUCCESS = "#15803D"

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("课堂录屏 PPT 换页检测")
        self.root.geometry("1040x920")
        self.root.minsize(900, 700)
        self.root.configure(background=self.BACKGROUND)

        self.events: queue.Queue[tuple[object, ...]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.running = False
        self.last_result_path: Path | None = None
        self.last_output_dir: Path | None = None
        self.screenshot_by_item: dict[str, Path] = {}
        self.base_config = DetectorConfig()

        self.video_var = tk.StringVar()
        self.output_var = tk.StringVar(value=str(Path.cwd() / "output"))
        default_config = Path.cwd() / "config" / "default.json"
        self.config_var = tk.StringVar(
            value=str(default_config) if default_config.is_file() else ""
        )
        self.video_id_var = tk.StringVar()
        self.ffmpeg_var = tk.StringVar()
        self.ffprobe_var = tk.StringVar()
        self.parameter_vars: dict[str, tk.StringVar] = {}
        self.status_var = tk.StringVar(value="请选择一段课堂录屏")
        self.progress_var = tk.DoubleVar(value=0)

        self._configure_styles()
        self._build_interface()
        self._apply_config_to_fields(self._load_selected_config(show_error=False))
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
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
            foreground=self.TEXT,
            font=("Microsoft YaHei UI", 22, "bold"),
        )
        style.configure(
            "Subtitle.TLabel",
            background=self.BACKGROUND,
            foreground=self.MUTED,
            font=("Microsoft YaHei UI", 10),
        )
        style.configure(
            "CardTitle.TLabel",
            background=self.CARD,
            foreground=self.TEXT,
            font=("Microsoft YaHei UI", 12, "bold"),
        )
        style.configure(
            "CardText.TLabel",
            background=self.CARD,
            foreground=self.TEXT,
        )
        style.configure(
            "Help.TLabel",
            background=self.CARD,
            foreground=self.MUTED,
            font=("Microsoft YaHei UI", 9),
        )
        style.configure(
            "Primary.TButton",
            background=self.PRIMARY,
            foreground="white",
            padding=(18, 10),
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        style.map(
            "Primary.TButton",
            background=[
                ("active", self.PRIMARY_DARK),
                ("disabled", "#A8B9D6"),
            ],
        )
        style.configure("Secondary.TButton", padding=(12, 8))
        style.configure(
            "App.Horizontal.TProgressbar",
            background=self.PRIMARY,
            troughcolor="#E7EDF6",
            bordercolor="#E7EDF6",
            lightcolor=self.PRIMARY,
            darkcolor=self.PRIMARY,
        )
        style.configure(
            "Treeview",
            rowheight=30,
            background="white",
            fieldbackground="white",
            foreground=self.TEXT,
        )
        style.configure(
            "Treeview.Heading",
            font=("Microsoft YaHei UI", 9, "bold"),
            foreground=self.TEXT,
        )

    def _build_interface(self) -> None:
        main = ttk.Frame(self.root, style="App.TFrame", padding=(24, 16))
        main.pack(fill="both", expand=True)

        header = ttk.Frame(main, style="App.TFrame")
        header.pack(fill="x", pady=(0, 12))
        ttk.Label(
            header,
            text="课堂录屏 PPT 换页检测",
            style="Title.TLabel",
        ).pack(anchor="w")
        ttk.Label(
            header,
            text="选择视频后自动生成换页时间表、代表截图和低置信度复核记录",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        notebook = ttk.Notebook(main)
        notebook.pack(fill="both", expand=True)
        process_tab = ttk.Frame(notebook, style="App.TFrame", padding=(2, 10))
        settings_tab = ttk.Frame(notebook, style="App.TFrame", padding=(2, 10))
        notebook.add(process_tab, text="  视频处理  ")
        notebook.add(settings_tab, text="  参数设置  ")

        self._build_process_tab(process_tab)
        self._build_settings_tab(settings_tab)

    def _build_process_tab(self, parent: ttk.Frame) -> None:
        input_card = ttk.Frame(parent, style="Card.TFrame", padding=14)
        input_card.pack(fill="x")
        ttk.Label(
            input_card,
            text="输入与输出",
            style="CardTitle.TLabel",
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
        input_card.columnconfigure(1, weight=1)

        self._path_row(
            input_card,
            row=1,
            label="课堂录屏",
            variable=self.video_var,
            command=self._choose_video,
            button_text="选择视频",
        )
        self._path_row(
            input_card,
            row=2,
            label="输出目录",
            variable=self.output_var,
            command=self._choose_output,
            button_text="选择目录",
        )
        ttk.Label(
            input_card,
            text="视频 ID",
            style="CardText.TLabel",
        ).grid(row=3, column=0, sticky="w", padx=(0, 12), pady=4)
        ttk.Entry(input_card, textvariable=self.video_id_var).grid(
            row=3,
            column=1,
            sticky="ew",
            pady=4,
        )
        ttk.Label(
            input_card,
            text="可留空，默认使用视频文件名",
            style="Help.TLabel",
        ).grid(row=3, column=2, sticky="w", padx=(12, 0), pady=4)

        action_row = ttk.Frame(input_card, style="Card.TFrame")
        action_row.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        self.start_button = ttk.Button(
            action_row,
            text="开始检测",
            style="Primary.TButton",
            command=self._start_detection,
        )
        self.start_button.pack(side="left")
        self.open_output_button = ttk.Button(
            action_row,
            text="打开输出目录",
            style="Secondary.TButton",
            command=self._open_output,
            state="disabled",
        )
        self.open_output_button.pack(side="left", padx=(10, 0))
        self.open_result_button = ttk.Button(
            action_row,
            text="打开结果 JSON",
            style="Secondary.TButton",
            command=self._open_result,
            state="disabled",
        )
        self.open_result_button.pack(side="left", padx=(8, 0))

        status_card = ttk.Frame(parent, style="Card.TFrame", padding=14)
        status_card.pack(fill="x", pady=(14, 0))
        status_header = ttk.Frame(status_card, style="Card.TFrame")
        status_header.pack(fill="x")
        ttk.Label(
            status_header,
            text="处理状态",
            style="CardTitle.TLabel",
        ).pack(side="left")
        self.status_label = ttk.Label(
            status_header,
            textvariable=self.status_var,
            style="Help.TLabel",
        )
        self.status_label.pack(side="right")
        self.progress = ttk.Progressbar(
            status_card,
            variable=self.progress_var,
            maximum=100,
            mode="determinate",
            style="App.Horizontal.TProgressbar",
        )
        self.progress.pack(fill="x", pady=(12, 10))
        self.log = ScrolledText(
            status_card,
            height=3,
            wrap="word",
            relief="flat",
            background="#F8FAFC",
            foreground=self.TEXT,
            font=("Microsoft YaHei UI", 9),
            padx=10,
            pady=8,
            state="disabled",
        )
        self.log.pack(fill="x")

        result_card = ttk.Frame(parent, style="Card.TFrame", padding=14)
        result_card.pack(fill="both", expand=True, pady=(14, 0))
        ttk.Label(
            result_card,
            text="检测结果",
            style="CardTitle.TLabel",
        ).pack(anchor="w", pady=(0, 10))
        columns = ("page", "start", "end", "confidence", "note")
        self.result_tree = ttk.Treeview(
            result_card,
            columns=columns,
            show="headings",
            selectmode="browse",
            height=5,
        )
        headings = {
            "page": "页码",
            "start": "开始时间",
            "end": "结束时间",
            "confidence": "置信度",
            "note": "说明",
        }
        widths = {
            "page": 70,
            "start": 120,
            "end": 120,
            "confidence": 90,
            "note": 460,
        }
        for column in columns:
            self.result_tree.heading(column, text=headings[column])
            self.result_tree.column(
                column,
                width=widths[column],
                minwidth=60,
                anchor="center" if column != "note" else "w",
            )
        scrollbar = ttk.Scrollbar(
            result_card,
            orient="vertical",
            command=self.result_tree.yview,
        )
        self.result_tree.configure(yscrollcommand=scrollbar.set)
        self.result_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.result_tree.bind("<Double-1>", self._open_selected_screenshot)

    def _build_settings_tab(self, parent: ttk.Frame) -> None:
        config_card = ttk.Frame(parent, style="Card.TFrame", padding=18)
        config_card.pack(fill="x")
        config_card.columnconfigure(1, weight=1)
        ttk.Label(
            config_card,
            text="配置来源",
            style="CardTitle.TLabel",
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 12))
        self._path_row(
            config_card,
            row=1,
            label="配置文件",
            variable=self.config_var,
            command=self._choose_config,
            button_text="选择 JSON",
        )
        ttk.Button(
            config_card,
            text="重新载入配置",
            command=self._reload_config,
        ).grid(row=2, column=1, sticky="w", pady=(8, 0))
        ttk.Button(
            config_card,
            text="恢复内置默认值",
            command=self._reset_config,
        ).grid(row=2, column=1, sticky="w", padx=(120, 0), pady=(8, 0))

        parameter_card = ttk.Frame(parent, style="Card.TFrame", padding=18)
        parameter_card.pack(fill="both", expand=True, pady=(14, 0))
        ttk.Label(
            parameter_card,
            text="检测参数",
            style="CardTitle.TLabel",
        ).grid(row=0, column=0, columnspan=6, sticky="w", pady=(0, 12))
        parameter_card.columnconfigure(1, weight=1)
        parameter_card.columnconfigure(4, weight=1)

        for index, (key, label, help_text, _) in enumerate(PARAMETER_FIELDS):
            group = index % 2
            row = index // 2 + 1
            base_column = group * 3
            ttk.Label(
                parameter_card,
                text=label,
                style="CardText.TLabel",
            ).grid(
                row=row,
                column=base_column,
                sticky="w",
                padx=(0, 8),
                pady=8,
            )
            variable = tk.StringVar()
            self.parameter_vars[key] = variable
            ttk.Entry(
                parameter_card,
                textvariable=variable,
                width=14,
            ).grid(row=row, column=base_column + 1, sticky="ew", pady=8)
            ttk.Label(
                parameter_card,
                text=help_text,
                style="Help.TLabel",
            ).grid(
                row=row,
                column=base_column + 2,
                sticky="w",
                padx=(8, 18 if group == 0 else 0),
                pady=8,
            )

        binary_row = len(PARAMETER_FIELDS) // 2 + 2
        ttk.Separator(parameter_card).grid(
            row=binary_row,
            column=0,
            columnspan=6,
            sticky="ew",
            pady=(12, 8),
        )
        self._path_row(
            parameter_card,
            row=binary_row + 1,
            label="FFmpeg",
            variable=self.ffmpeg_var,
            command=lambda: self._choose_binary(self.ffmpeg_var),
            button_text="选择程序",
            column_span=4,
        )
        self._path_row(
            parameter_card,
            row=binary_row + 2,
            label="FFprobe",
            variable=self.ffprobe_var,
            command=lambda: self._choose_binary(self.ffprobe_var),
            button_text="选择程序",
            column_span=4,
        )
        ttk.Label(
            parameter_card,
            text="留空时自动从系统 PATH 查找 FFmpeg；参数修改只影响本次运行。",
            style="Help.TLabel",
        ).grid(
            row=binary_row + 3,
            column=0,
            columnspan=6,
            sticky="w",
            pady=(10, 0),
        )

    @staticmethod
    def _path_row(
        parent: ttk.Frame,
        *,
        row: int,
        label: str,
        variable: tk.StringVar,
        command: Callable[[], None],
        button_text: str,
        column_span: int = 1,
    ) -> None:
        ttk.Label(parent, text=label, style="CardText.TLabel").grid(
            row=row,
            column=0,
            sticky="w",
            padx=(0, 12),
            pady=4,
        )
        ttk.Entry(parent, textvariable=variable).grid(
            row=row,
            column=1,
            columnspan=column_span,
            sticky="ew",
            pady=4,
        )
        ttk.Button(parent, text=button_text, command=command).grid(
            row=row,
            column=column_span + 1,
            sticky="e",
            padx=(10, 0),
            pady=4,
        )

    def _choose_video(self) -> None:
        path = filedialog.askopenfilename(
            title="选择课堂录屏",
            filetypes=[
                ("视频文件", "*.mp4 *.avi *.mov *.mkv *.wmv *.m4v"),
                ("所有文件", "*.*"),
            ],
        )
        if path:
            self.video_var.set(path)
            if not self.video_id_var.get().strip():
                self.video_id_var.set(Path(path).stem)

    def _choose_output(self) -> None:
        path = filedialog.askdirectory(title="选择输出目录")
        if path:
            self.output_var.set(path)

    def _choose_config(self) -> None:
        path = filedialog.askopenfilename(
            title="选择配置文件",
            filetypes=[("JSON 配置", "*.json"), ("所有文件", "*.*")],
        )
        if path:
            self.config_var.set(path)
            self._reload_config()

    @staticmethod
    def _choose_binary(variable: tk.StringVar) -> None:
        path = filedialog.askopenfilename(
            title="选择程序",
            filetypes=[("可执行程序", "*.exe"), ("所有文件", "*.*")],
        )
        if path:
            variable.set(path)

    def _load_selected_config(
        self,
        *,
        show_error: bool = True,
        fallback: bool = True,
    ) -> DetectorConfig:
        path_text = self.config_var.get().strip()
        if not path_text:
            return DetectorConfig()
        try:
            return DetectorConfig.from_file(path_text)
        except (OSError, ValueError) as exc:
            if show_error:
                messagebox.showerror("配置错误", str(exc), parent=self.root)
            if not fallback:
                raise
            return DetectorConfig()

    def _apply_config_to_fields(self, config: DetectorConfig) -> None:
        self.base_config = config
        values = config.to_dict()
        for key, variable in self.parameter_vars.items():
            variable.set(str(values[key]))
        self.ffmpeg_var.set(config.ffmpeg_path or "")
        self.ffprobe_var.set(config.ffprobe_path or "")

    def _reload_config(self) -> None:
        self._apply_config_to_fields(self._load_selected_config())
        self._append_log("已重新载入配置文件。")

    def _reset_config(self) -> None:
        self._apply_config_to_fields(DetectorConfig())
        self._append_log("已恢复内置默认参数。")

    def _start_detection(self) -> None:
        if self.running:
            return
        video_path = Path(self.video_var.get().strip())
        if not video_path.is_file():
            messagebox.showwarning(
                "请选择视频",
                "请选择一个存在的课堂录屏文件。",
                parent=self.root,
            )
            return
        output_text = self.output_var.get().strip()
        if not output_text:
            messagebox.showwarning(
                "请选择输出目录",
                "输出目录不能为空。",
                parent=self.root,
            )
            return
        try:
            base_config = self._load_selected_config(
                show_error=False,
                fallback=False,
            )
            config = config_from_gui_values(
                base_config,
                {key: variable.get() for key, variable in self.parameter_vars.items()},
                ffmpeg_path=self.ffmpeg_var.get(),
                ffprobe_path=self.ffprobe_var.get(),
            )
        except (OSError, ValueError) as exc:
            messagebox.showerror("参数错误", str(exc), parent=self.root)
            return

        output_root = Path(output_text)
        video_id = self.video_id_var.get().strip() or None
        self.running = True
        self.last_result_path = None
        self.last_output_dir = None
        self.start_button.configure(state="disabled", text="处理中…")
        self.open_output_button.configure(state="disabled")
        self.open_result_button.configure(state="disabled")
        self.progress_var.set(0)
        self.status_var.set("正在启动处理")
        self._clear_results()
        self._clear_log()
        self._append_log(f"输入视频：{video_path}")
        self._append_log(f"输出目录：{output_root}")

        self.worker = threading.Thread(
            target=self._run_worker,
            args=(video_path, output_root, video_id, config),
            daemon=True,
        )
        self.worker.start()

    def _run_worker(
        self,
        video_path: Path,
        output_root: Path,
        video_id: str | None,
        config: DetectorConfig,
    ) -> None:
        try:
            detector = VideoPageDetector(config)
            result = detector.run(
                video_path,
                output_root=output_root,
                video_id=video_id,
                progress_callback=lambda message, progress: self.events.put(
                    ("progress", message, progress)
                ),
            )
            self.events.put(("complete", result, output_root))
        except Exception as exc:  # GUI boundary: surface all worker failures
            self.events.put(("error", str(exc), traceback.format_exc()))

    def _poll_events(self) -> None:
        try:
            while True:
                event = self.events.get_nowait()
                event_type = event[0]
                if event_type == "progress":
                    _, message, progress = event
                    self._handle_progress(str(message), progress)
                elif event_type == "complete":
                    _, result, output_root = event
                    self._handle_complete(result, Path(output_root))
                elif event_type == "error":
                    _, message, details = event
                    self._handle_error(str(message), str(details))
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def _handle_progress(self, message: str, progress: object) -> None:
        self.status_var.set(message)
        if isinstance(progress, (int, float)):
            self.progress_var.set(max(0, min(100, float(progress) * 100)))
        self._append_log(message)

    def _handle_complete(self, result: object, output_root: Path) -> None:
        if not isinstance(result, dict):
            self._handle_error("处理结果格式不正确", repr(result))
            return
        self.running = False
        self.start_button.configure(state="normal", text="开始检测")
        self.progress_var.set(100)
        self.status_var.set("处理完成")
        video_id = str(result["video_id"])
        self.last_output_dir = output_root / video_id
        self.last_result_path = self.last_output_dir / "result.json"
        self.open_output_button.configure(state="normal")
        self.open_result_button.configure(state="normal")
        self._populate_results(result)
        page_count = len(result.get("pages", []))
        review_count = sum(
            page.get("confidence") != "high"
            for page in result.get("pages", [])
        )
        self._append_log(
            f"完成：检测到 {page_count} 个页面，其中 {review_count} 个建议复核。"
        )
        messagebox.showinfo(
            "处理完成",
            (
                f"共生成 {page_count} 个页面。\n"
                f"建议复核：{review_count} 个。\n\n"
                f"结果目录：\n{self.last_output_dir}"
            ),
            parent=self.root,
        )

    def _handle_error(self, message: str, details: str) -> None:
        self.running = False
        self.start_button.configure(state="normal", text="开始检测")
        self.status_var.set("处理失败")
        self._append_log(f"处理失败：{message}")
        self._append_log(details)
        messagebox.showerror(
            "处理失败",
            (
                f"{message}\n\n"
                "如果提示找不到 FFmpeg，请在“参数设置”中选择 "
                "ffmpeg.exe 和 ffprobe.exe。"
            ),
            parent=self.root,
        )

    def _populate_results(self, result: dict[str, object]) -> None:
        self._clear_results()
        pages = result.get("pages", [])
        if not isinstance(pages, list):
            return
        for index, page in enumerate(pages):
            if not isinstance(page, dict):
                continue
            item = self.result_tree.insert(
                "",
                "end",
                values=(
                    page.get("page_id", ""),
                    format_seconds(float(page.get("start_sec", 0))),
                    format_seconds(float(page.get("end_sec", 0))),
                    page.get("confidence", ""),
                    page.get("note", ""),
                ),
                tags=("even" if index % 2 == 0 else "odd",),
            )
            screenshot = page.get("screenshot_path")
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
        self.log.insert("end", f"{message}\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _open_output(self) -> None:
        if self.last_output_dir and self.last_output_dir.exists():
            open_external(self.last_output_dir)

    def _open_result(self) -> None:
        if self.last_result_path and self.last_result_path.is_file():
            open_external(self.last_result_path)

    def _open_selected_screenshot(self, _: tk.Event[tk.Misc]) -> None:
        selection = self.result_tree.selection()
        if not selection:
            return
        path = self.screenshot_by_item.get(selection[0])
        if path and path.is_file():
            open_external(path)

    def _on_close(self) -> None:
        if self.running and not messagebox.askyesno(
            "确认退出",
            "视频仍在处理中，退出会中断本次操作。确定退出吗？",
            parent=self.root,
        ):
            return
        self.root.destroy()


def main() -> int:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"Unable to start GUI: {exc}", file=sys.stderr)
        return 2
    DetectorApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
