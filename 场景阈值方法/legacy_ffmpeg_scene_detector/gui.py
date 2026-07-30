from __future__ import annotations

import os
import queue
import shutil
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


FieldDefinition = tuple[str, str, str, Callable[[str], object]]

COMMON_FIELDS: tuple[FieldDefinition, ...] = (
    (
        "scene_threshold",
        "FFmpeg 场景阈值",
        "越小越敏感、候选越多；旧方法最常调整的参数",
        float,
    ),
    (
        "scene_threshold_floor",
        "自动降阈值下限",
        "候选不足时允许降低到的最小阈值",
        float,
    ),
    (
        "minimum_scene_candidates_per_minute",
        "每分钟最少候选数",
        "候选不足时触发自动降低阈值",
        float,
    ),
    (
        "high_change_ratio",
        "高置信变化比例",
        "超过该分块比例直接判定为高置信换页",
        float,
    ),
    (
        "low_change_ratio",
        "低置信变化比例",
        "超过该值但未到高阈值时保留为低置信换页",
        float,
    ),
    (
        "min_page_duration_sec",
        "最短页面时长",
        "短于该秒数的页面会被合并",
        float,
    ),
    (
        "duplicate_hash_distance",
        "全局哈希去重距离",
        "仅作为初筛；还会检查内容分块变化",
        int,
    ),
    (
        "duplicate_changed_block_ratio",
        "去重分块变化上限",
        "低于该比例且全局相似才视为重复，默认 0.25",
        float,
    ),
)

ADVANCED_FIELDS: tuple[FieldDefinition, ...] = (
    ("grid_columns", "分块列数", "画面横向分成几块", int),
    ("grid_rows", "分块行数", "画面纵向分成几块", int),
    (
        "block_hash_distance",
        "分块差异距离",
        "单个分块达到该距离才算发生变化",
        int,
    ),
    (
        "comparison_offset_sec",
        "换页前比较偏移",
        "从候选点前多少秒取对比帧",
        float,
    ),
    ("stable_delay_sec", "稳定帧等待", "候选点后等待多少秒", float),
    ("stable_window_sec", "稳定帧窗口", "搜索稳定画面的窗口长度", float),
    ("stable_sample_fps", "稳定窗口帧率", "窗口内每秒分析帧数", float),
    (
        "stable_diff_threshold",
        "稳定帧差阈值",
        "相邻帧差低于该值认为画面稳定",
        float,
    ),
    (
        "no_ppt_min_duration_sec",
        "无 PPT 最短时长",
        "连续不稳定达到该秒数才记录",
        float,
    ),
    (
        "no_ppt_merge_gap_sec",
        "无 PPT 合并间隔",
        "相距较近的不稳定区间合并",
        float,
    ),
    ("analysis_width", "分析帧宽度", "仅用于检测，不决定最终视频尺寸", int),
    ("analysis_height", "分析帧高度", "仅用于检测，不决定最终视频尺寸", int),
    ("jpeg_quality", "JPEG 质量", "范围 1～100", int),
    ("screen_crop_left_ratio", "投影区左边距", "画面比例，例如 0.10", float),
    ("screen_crop_top_ratio", "投影区上边距", "画面比例，例如 0.02", float),
    ("screen_crop_right_ratio", "投影区右边距", "画面比例，例如 0.10", float),
    ("screen_crop_bottom_ratio", "投影区下边距", "画面比例，例如 0.10", float),
)

ALL_FIELDS = COMMON_FIELDS + ADVANCED_FIELDS


def format_seconds(seconds: float) -> str:
    total_milliseconds = int(round(float(seconds) * 1000))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs = remainder / 1000
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    return f"{minutes:02d}:{secs:06.3f}"


def discover_executable(name: str) -> str:
    discovered = shutil.which(name)
    if discovered:
        return discovered
    known = Path(r"C:\Program Files\FFmpeg\bin") / f"{name}.exe"
    return str(known) if known.is_file() else ""


def config_from_gui_values(
    base: DetectorConfig,
    values: Mapping[str, str],
    *,
    adaptive_scene_detection: bool,
    auto_detect_screen_crop: bool,
    crop_output_screenshots: bool,
    ffmpeg_path: str = "",
    ffprobe_path: str = "",
) -> DetectorConfig:
    converters = {key: converter for key, _, _, converter in ALL_FIELDS}
    overrides: dict[str, object] = {}
    for key, converter in converters.items():
        raw = values.get(key, "").strip()
        if not raw:
            raise ValueError(f"参数“{key}”不能为空")
        try:
            overrides[key] = converter(raw)
        except ValueError as exc:
            raise ValueError(f"参数“{key}”格式不正确：{raw}") from exc
    overrides.update(
        {
            "adaptive_scene_detection": adaptive_scene_detection,
            "auto_detect_screen_crop": auto_detect_screen_crop,
            "crop_output_screenshots": crop_output_screenshots,
            "ffmpeg_path": ffmpeg_path.strip() or None,
            "ffprobe_path": ffprobe_path.strip() or None,
        }
    )
    config = replace(base, **overrides)
    config.validate()
    return config


def open_path(path: Path) -> None:
    if sys.platform == "win32":
        os.startfile(path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


class LegacyDetectorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("旧版 FFmpeg 场景阈值检测器")
        self.root.geometry("1120x850")
        self.root.minsize(980, 720)

        config_path = Path(__file__).parent / "config" / "default.json"
        self.base_config = DetectorConfig.from_file(config_path)
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.parameter_vars = {
            key: tk.StringVar(value=str(getattr(self.base_config, key)))
            for key, _, _, _ in ALL_FIELDS
        }
        self.video_var = tk.StringVar()
        self.output_var = tk.StringVar(
            value=str(Path.cwd() / "legacy_output")
        )
        self.ffmpeg_var = tk.StringVar(value=discover_executable("ffmpeg"))
        self.ffprobe_var = tk.StringVar(value=discover_executable("ffprobe"))
        self.adaptive_var = tk.BooleanVar(
            value=self.base_config.adaptive_scene_detection
        )
        self.auto_crop_var = tk.BooleanVar(
            value=self.base_config.auto_detect_screen_crop
        )
        self.crop_output_var = tk.BooleanVar(
            value=self.base_config.crop_output_screenshots
        )
        self.status_var = tk.StringVar(value="请选择视频后开始处理")
        self.last_output_dir: Path | None = None
        self.screenshot_by_item: dict[str, Path] = {}

        self._build_ui()
        self.root.after(100, self._poll_events)

    def _build_ui(self) -> None:
        style = ttk.Style(self.root)
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 18, "bold"))
        style.configure("Warning.TLabel", foreground="#9a5b00")

        container = ttk.Frame(self.root, padding=14)
        container.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            container,
            text="旧版 FFmpeg 场景阈值检测器",
            style="Title.TLabel",
        ).pack(anchor=tk.W)
        ttk.Label(
            container,
            text=(
                "这是独立归档的旧算法。它依赖场景阈值，"
                "不同视频可能需要调整参数。"
            ),
            style="Warning.TLabel",
        ).pack(anchor=tk.W, pady=(2, 10))

        source_frame = ttk.LabelFrame(container, text="文件设置", padding=10)
        source_frame.pack(fill=tk.X)
        source_frame.columnconfigure(1, weight=1)
        self._path_row(
            source_frame,
            0,
            "输入视频",
            self.video_var,
            self._choose_video,
            "选择视频",
        )
        self._path_row(
            source_frame,
            1,
            "输出目录",
            self.output_var,
            self._choose_output,
            "选择目录",
        )

        self.notebook = ttk.Notebook(container)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=10)

        common_tab = ttk.Frame(self.notebook, padding=12)
        advanced_tab = ttk.Frame(self.notebook, padding=12)
        tools_tab = ttk.Frame(self.notebook, padding=12)
        results_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(common_tab, text="常用检测参数")
        self.notebook.add(advanced_tab, text="高级参数")
        self.notebook.add(tools_tab, text="FFmpeg 与输出")
        self.notebook.add(results_tab, text="处理结果")

        self._build_field_table(common_tab, COMMON_FIELDS, columns=1)
        ttk.Checkbutton(
            common_tab,
            text="候选点不足时自动降低 FFmpeg 场景阈值",
            variable=self.adaptive_var,
        ).pack(anchor=tk.W, pady=(12, 0))

        self._build_field_table(advanced_tab, ADVANCED_FIELDS, columns=2)
        self._build_tools_tab(tools_tab)
        self._build_results_tab(results_tab)

        footer = ttk.Frame(container)
        footer.pack(fill=tk.X)
        footer.columnconfigure(0, weight=1)
        ttk.Label(footer, textvariable=self.status_var).grid(
            row=0, column=0, sticky=tk.W
        )
        self.progress = ttk.Progressbar(
            footer,
            mode="determinate",
            maximum=100,
            length=360,
        )
        self.progress.grid(row=1, column=0, sticky=tk.EW, pady=(5, 0))
        button_frame = ttk.Frame(footer)
        button_frame.grid(row=0, column=1, rowspan=2, padx=(12, 0))
        self.reset_button = ttk.Button(
            button_frame,
            text="恢复默认参数",
            command=self._reset_defaults,
        )
        self.reset_button.pack(side=tk.LEFT)
        self.open_button = ttk.Button(
            button_frame,
            text="打开输出目录",
            command=self._open_output,
            state=tk.DISABLED,
        )
        self.open_button.pack(side=tk.LEFT, padx=8)
        self.start_button = ttk.Button(
            button_frame,
            text="开始旧版检测",
            command=self._start,
        )
        self.start_button.pack(side=tk.LEFT)

    @staticmethod
    def _path_row(
        parent: ttk.LabelFrame,
        row: int,
        label: str,
        variable: tk.StringVar,
        command: Callable[[], None],
        button_text: str,
    ) -> None:
        ttk.Label(parent, text=label, width=10).grid(
            row=row, column=0, sticky=tk.W, pady=4
        )
        ttk.Entry(parent, textvariable=variable).grid(
            row=row, column=1, sticky=tk.EW, padx=8, pady=4
        )
        ttk.Button(parent, text=button_text, command=command).grid(
            row=row, column=2, pady=4
        )

    def _build_field_table(
        self,
        parent: ttk.Frame,
        definitions: tuple[FieldDefinition, ...],
        *,
        columns: int,
    ) -> None:
        holder = ttk.Frame(parent)
        holder.pack(fill=tk.BOTH, expand=True)
        for column in range(columns):
            holder.columnconfigure(column * 3 + 2, weight=1)
        rows_per_column = (len(definitions) + columns - 1) // columns
        for index, (key, label, hint, _) in enumerate(definitions):
            column_group = index // rows_per_column
            row = index % rows_per_column
            column = column_group * 3
            ttk.Label(holder, text=label, width=17).grid(
                row=row, column=column, sticky=tk.W, pady=6
            )
            ttk.Entry(
                holder,
                textvariable=self.parameter_vars[key],
                width=12,
            ).grid(row=row, column=column + 1, sticky=tk.W, padx=(4, 8))
            ttk.Label(
                holder,
                text=hint,
                foreground="#666666",
                wraplength=245,
            ).grid(
                row=row,
                column=column + 2,
                sticky=tk.W,
                padx=(0, 18),
            )

    def _build_tools_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)
        self._path_row(
            parent,
            0,
            "FFmpeg",
            self.ffmpeg_var,
            lambda: self._choose_executable(self.ffmpeg_var),
            "选择文件",
        )
        self._path_row(
            parent,
            1,
            "FFprobe",
            self.ffprobe_var,
            lambda: self._choose_executable(self.ffprobe_var),
            "选择文件",
        )
        ttk.Checkbutton(
            parent,
            text="自动识别投影区域（推荐）",
            variable=self.auto_crop_var,
        ).grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=(14, 6))
        ttk.Checkbutton(
            parent,
            text="代表截图只保留设定的投影区域",
            variable=self.crop_output_var,
        ).grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=6)
        ttk.Label(
            parent,
            text=(
                "路径为空时会从系统 PATH 查找。当前电脑已安装到 "
                r"C:\Program Files\FFmpeg\bin 时，界面会自动识别。"
            ),
            foreground="#666666",
            wraplength=850,
        ).grid(row=4, column=0, columnspan=3, sticky=tk.W)

    def _build_results_tab(self, parent: ttk.Frame) -> None:
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)
        columns = ("page", "start", "end", "confidence", "screenshot")
        self.result_tree = ttk.Treeview(
            parent,
            columns=columns,
            show="headings",
            height=12,
        )
        headings = {
            "page": "页码",
            "start": "开始时间",
            "end": "结束时间",
            "confidence": "置信度",
            "screenshot": "截图",
        }
        widths = {
            "page": 60,
            "start": 105,
            "end": 105,
            "confidence": 80,
            "screenshot": 580,
        }
        for column in columns:
            self.result_tree.heading(column, text=headings[column])
            self.result_tree.column(
                column,
                width=widths[column],
                anchor=tk.W if column == "screenshot" else tk.CENTER,
            )
        self.result_tree.grid(row=0, column=0, sticky=tk.NSEW)
        scrollbar = ttk.Scrollbar(
            parent,
            orient=tk.VERTICAL,
            command=self.result_tree.yview,
        )
        scrollbar.grid(row=0, column=1, sticky=tk.NS)
        self.result_tree.configure(yscrollcommand=scrollbar.set)
        self.result_tree.bind("<Double-1>", self._open_selected_screenshot)

        ttk.Label(parent, text="运行日志").grid(
            row=1, column=0, sticky=tk.W, pady=(8, 3)
        )
        self.log = ScrolledText(parent, height=7, state=tk.DISABLED)
        self.log.grid(row=2, column=0, columnspan=2, sticky=tk.EW)

    def _choose_video(self) -> None:
        selected = filedialog.askopenfilename(
            title="选择课堂视频",
            filetypes=[
                ("视频文件", "*.mp4 *.mkv *.avi *.mov *.flv *.wmv"),
                ("所有文件", "*.*"),
            ],
        )
        if selected:
            self.video_var.set(selected)

    def _choose_output(self) -> None:
        selected = filedialog.askdirectory(title="选择输出目录")
        if selected:
            self.output_var.set(selected)

    @staticmethod
    def _choose_executable(variable: tk.StringVar) -> None:
        selected = filedialog.askopenfilename(
            title="选择程序",
            filetypes=[("可执行文件", "*.exe"), ("所有文件", "*.*")],
        )
        if selected:
            variable.set(selected)

    def _reset_defaults(self) -> None:
        for key, _, _, _ in ALL_FIELDS:
            self.parameter_vars[key].set(str(getattr(self.base_config, key)))
        self.adaptive_var.set(self.base_config.adaptive_scene_detection)
        self.auto_crop_var.set(self.base_config.auto_detect_screen_crop)
        self.crop_output_var.set(self.base_config.crop_output_screenshots)
        self.status_var.set("已恢复旧版默认参数")

    def _current_config(self) -> DetectorConfig:
        return config_from_gui_values(
            self.base_config,
            {key: variable.get() for key, variable in self.parameter_vars.items()},
            adaptive_scene_detection=self.adaptive_var.get(),
            auto_detect_screen_crop=self.auto_crop_var.get(),
            crop_output_screenshots=self.crop_output_var.get(),
            ffmpeg_path=self.ffmpeg_var.get(),
            ffprobe_path=self.ffprobe_var.get(),
        )

    def _start(self) -> None:
        try:
            video = Path(self.video_var.get().strip())
            if not video.is_file():
                raise ValueError("请选择存在的视频文件")
            output_text = self.output_var.get().strip()
            if not output_text:
                raise ValueError("请选择输出目录")
            output = Path(output_text)
            config = self._current_config()
        except ValueError as exc:
            messagebox.showerror("参数错误", str(exc), parent=self.root)
            return

        self._clear_results()
        self._append_log(f"输入视频：{video}")
        self._append_log(f"输出目录：{output}")
        self._append_log(
            f"场景阈值：{config.scene_threshold:g}；"
            f"自动降阈值：{'开启' if config.adaptive_scene_detection else '关闭'}"
        )
        self._set_running(True)
        self.notebook.select(3)
        worker = threading.Thread(
            target=self._run_worker,
            args=(video, output, config),
            daemon=True,
        )
        worker.start()

    def _run_worker(
        self,
        video: Path,
        output: Path,
        config: DetectorConfig,
    ) -> None:
        try:
            result = VideoPageDetector(config).run(
                video,
                output_root=output,
                progress_callback=lambda message, progress: self.events.put(
                    ("progress", (message, progress))
                ),
            )
            self.events.put(("done", (result, output)))
        except Exception as exc:
            self.events.put(("error", (str(exc), traceback.format_exc())))

    def _poll_events(self) -> None:
        while True:
            try:
                event, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if event == "progress":
                message, progress = payload  # type: ignore[misc]
                self.status_var.set(str(message))
                self._append_log(str(message))
                if progress is not None:
                    self.progress["value"] = max(
                        0.0, min(100.0, float(progress) * 100.0)
                    )
            elif event == "done":
                result, output = payload  # type: ignore[misc]
                self._finish_success(result, output)
            else:
                message, details = payload  # type: ignore[misc]
                self._finish_error(str(message), str(details))
        self.root.after(100, self._poll_events)

    def _finish_success(
        self,
        result: Mapping[str, object],
        output: Path,
    ) -> None:
        pages = result.get("pages", [])
        if not isinstance(pages, list):
            pages = []
        self.screenshot_by_item.clear()
        for page in pages:
            if not isinstance(page, dict):
                continue
            screenshot = Path(str(page.get("screenshot_path", "")))
            item = self.result_tree.insert(
                "",
                tk.END,
                values=(
                    page.get("page_id", ""),
                    format_seconds(float(page.get("start_sec", 0.0))),
                    format_seconds(float(page.get("end_sec", 0.0))),
                    "高" if page.get("confidence") == "high" else "低",
                    str(screenshot),
                ),
            )
            self.screenshot_by_item[item] = screenshot
        video_id = str(result.get("video_id", ""))
        self.last_output_dir = output / video_id
        self.progress["value"] = 100
        self.status_var.set(f"处理完成，共输出 {len(pages)} 个页面")
        self._append_log(self.status_var.get())
        self._set_running(False)
        self.open_button.configure(state=tk.NORMAL)
        messagebox.showinfo(
            "处理完成",
            f"旧版检测完成，共输出 {len(pages)} 个页面。",
            parent=self.root,
        )

    def _finish_error(self, message: str, details: str) -> None:
        self.status_var.set("处理失败")
        self._append_log(details)
        self._set_running(False)
        messagebox.showerror("处理失败", message, parent=self.root)

    def _set_running(self, running: bool) -> None:
        self.start_button.configure(state=tk.DISABLED if running else tk.NORMAL)
        self.reset_button.configure(state=tk.DISABLED if running else tk.NORMAL)
        if running:
            self.progress["value"] = 0
            self.open_button.configure(state=tk.DISABLED)

    def _clear_results(self) -> None:
        for item in self.result_tree.get_children():
            self.result_tree.delete(item)
        self.screenshot_by_item.clear()
        self.last_output_dir = None
        self.log.configure(state=tk.NORMAL)
        self.log.delete("1.0", tk.END)
        self.log.configure(state=tk.DISABLED)

    def _append_log(self, message: str) -> None:
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, message.rstrip() + "\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def _open_selected_screenshot(self, _: tk.Event[tk.Misc]) -> None:
        selection = self.result_tree.selection()
        if not selection:
            return
        screenshot = self.screenshot_by_item.get(selection[0])
        if screenshot and screenshot.is_file():
            open_path(screenshot)

    def _open_output(self) -> None:
        if self.last_output_dir and self.last_output_dir.is_dir():
            open_path(self.last_output_dir)


def main() -> int:
    root = tk.Tk()
    LegacyDetectorApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
