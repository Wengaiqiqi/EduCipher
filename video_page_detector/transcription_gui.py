from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import traceback
from dataclasses import replace
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from .transcription import (
    TranscriptionConfig,
    format_timestamp,
    transcribe_video_pages,
)


def open_external(path: Path) -> None:
    target = str(path.resolve())
    if sys.platform == "win32":
        os.startfile(target)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", target])
    else:
        subprocess.Popen(["xdg-open", target])


class TranscriptionApp:
    BACKGROUND = "#F4F7FB"
    CARD = "#FFFFFF"
    PRIMARY = "#2563EB"
    TEXT = "#172033"
    MUTED = "#64748B"

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("PPT 逐页语音转文字")
        self.root.geometry("980x760")
        self.root.minsize(850, 650)
        self.root.configure(background=self.BACKGROUND)

        self.events: queue.Queue[tuple[object, ...]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.running = False
        self.last_json_path: Path | None = None
        self.last_markdown_path: Path | None = None

        self.video_var = tk.StringVar()
        self.result_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.config_var = tk.StringVar(
            value=str(Path.cwd() / "config" / "transcription.json")
        )
        self.model_var = tk.StringVar(value="small")
        self.hotwords_var = tk.StringVar(
            value=(
                "刚体 转动惯量 转动定律 角动量 "
                "角动量守恒 动能定理 质点"
            )
        )
        self.status_var = tk.StringVar(value="请选择视频和 PPT 检测结果")
        self.progress_var = tk.DoubleVar(value=0)

        self._configure_styles()
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
            foreground=self.TEXT,
            font=("Microsoft YaHei UI", 22, "bold"),
        )
        style.configure(
            "Subtitle.TLabel",
            background=self.BACKGROUND,
            foreground=self.MUTED,
        )
        style.configure(
            "CardTitle.TLabel",
            background=self.CARD,
            foreground=self.TEXT,
            font=("Microsoft YaHei UI", 12, "bold"),
        )
        style.configure("CardText.TLabel", background=self.CARD)
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
        style.configure("Secondary.TButton", padding=(12, 8))
        style.configure("Treeview", rowheight=34)
        style.configure(
            "Treeview.Heading",
            font=("Microsoft YaHei UI", 9, "bold"),
        )

    def _build_interface(self) -> None:
        main = ttk.Frame(self.root, style="App.TFrame", padding=(24, 18))
        main.pack(fill="both", expand=True)
        ttk.Label(
            main,
            text="PPT 逐页语音转文字",
            style="Title.TLabel",
        ).pack(anchor="w")
        ttk.Label(
            main,
            text=(
                "保留视频中的全部讲话和口头语，不区分说话人，"
                "按每页 PPT 时间自动整理；不会保留音频文件"
            ),
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(4, 14))

        input_card = ttk.Frame(main, style="Card.TFrame", padding=16)
        input_card.pack(fill="x")
        input_card.columnconfigure(1, weight=1)
        ttk.Label(
            input_card,
            text="输入与输出",
            style="CardTitle.TLabel",
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
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
            "PPT结果",
            self.result_var,
            self._choose_result,
            "选择 result.json",
        )
        self._path_row(
            input_card,
            3,
            "文字目录",
            self.output_var,
            self._choose_output,
            "选择目录",
        )

        ttk.Label(
            input_card,
            text="识别模型",
            style="CardText.TLabel",
        ).grid(row=4, column=0, sticky="w", padx=(0, 12), pady=5)
        model_box = ttk.Combobox(
            input_card,
            textvariable=self.model_var,
            values=("tiny", "base", "small", "medium", "large-v3"),
            state="readonly",
            width=18,
        )
        model_box.grid(row=4, column=1, sticky="w", pady=5)
        ttk.Label(
            input_card,
            text="推荐 small；模型越大越准确，但首次下载和CPU处理越慢",
            style="Help.TLabel",
        ).grid(row=4, column=2, sticky="w", padx=(12, 0), pady=5)

        ttk.Label(
            input_card,
            text="专业词汇",
            style="CardText.TLabel",
        ).grid(row=5, column=0, sticky="w", padx=(0, 12), pady=5)
        ttk.Entry(
            input_card,
            textvariable=self.hotwords_var,
        ).grid(row=5, column=1, sticky="ew", pady=5)
        ttk.Label(
            input_card,
            text="用空格分隔，可按课程内容增删；用于提高术语识别率",
            style="Help.TLabel",
        ).grid(row=5, column=2, sticky="w", padx=(12, 0), pady=5)

        action_row = ttk.Frame(input_card, style="Card.TFrame")
        action_row.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        self.start_button = ttk.Button(
            action_row,
            text="开始转成逐页文字",
            style="Primary.TButton",
            command=self._start,
        )
        self.start_button.pack(side="left")
        self.open_markdown_button = ttk.Button(
            action_row,
            text="打开逐页文字",
            style="Secondary.TButton",
            command=self._open_markdown,
            state="disabled",
        )
        self.open_markdown_button.pack(side="left", padx=(10, 0))
        self.open_output_button = ttk.Button(
            action_row,
            text="打开输出目录",
            style="Secondary.TButton",
            command=self._open_output,
            state="disabled",
        )
        self.open_output_button.pack(side="left", padx=(8, 0))

        status_card = ttk.Frame(main, style="Card.TFrame", padding=16)
        status_card.pack(fill="x", pady=(14, 0))
        status_header = ttk.Frame(status_card, style="Card.TFrame")
        status_header.pack(fill="x")
        ttk.Label(
            status_header,
            text="处理状态",
            style="CardTitle.TLabel",
        ).pack(side="left")
        ttk.Label(
            status_header,
            textvariable=self.status_var,
            style="Help.TLabel",
        ).pack(side="right")
        self.progress = ttk.Progressbar(
            status_card,
            variable=self.progress_var,
            maximum=100,
        )
        self.progress.pack(fill="x", pady=(12, 10))
        self.log = ScrolledText(
            status_card,
            height=4,
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

        result_card = ttk.Frame(main, style="Card.TFrame", padding=16)
        result_card.pack(fill="both", expand=True, pady=(14, 0))
        ttk.Label(
            result_card,
            text="逐页文字预览",
            style="CardTitle.TLabel",
        ).pack(anchor="w", pady=(0, 8))
        columns = ("page", "time", "text")
        self.result_tree = ttk.Treeview(
            result_card,
            columns=columns,
            show="headings",
            height=8,
        )
        self.result_tree.heading("page", text="页码")
        self.result_tree.heading("time", text="PPT时间")
        self.result_tree.heading("text", text="讲话文字")
        self.result_tree.column("page", width=70, anchor="center")
        self.result_tree.column("time", width=220, anchor="center")
        self.result_tree.column("text", width=590, anchor="w")
        scrollbar = ttk.Scrollbar(
            result_card,
            orient="vertical",
            command=self.result_tree.yview,
        )
        self.result_tree.configure(yscrollcommand=scrollbar.set)
        self.result_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _path_row(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.StringVar,
        command: object,
        button_text: str,
    ) -> None:
        ttk.Label(
            parent,
            text=label,
            style="CardText.TLabel",
        ).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=5)
        ttk.Entry(parent, textvariable=variable).grid(
            row=row,
            column=1,
            sticky="ew",
            pady=5,
        )
        ttk.Button(
            parent,
            text=button_text,
            command=command,
        ).grid(row=row, column=2, sticky="ew", padx=(12, 0), pady=5)

    def _choose_video(self) -> None:
        selected = filedialog.askopenfilename(
            title="选择课堂视频",
            filetypes=[
                ("视频文件", "*.mp4 *.mkv *.mov *.avi *.webm"),
                ("所有文件", "*.*"),
            ],
        )
        if not selected:
            return
        self.video_var.set(selected)
        stem = Path(selected).stem
        candidates = [
            Path.cwd() / "output" / stem / "result.json",
            Path.cwd() / "场景阈值方法" / "legacy_output" / stem / "result.json",
        ]
        if not self.result_var.get().strip():
            for candidate in candidates:
                if candidate.is_file():
                    self.result_var.set(str(candidate))
                    self.output_var.set(str(candidate.parent))
                    break

    def _choose_result(self) -> None:
        selected = filedialog.askopenfilename(
            title="选择 PPT 检测 result.json",
            filetypes=[("JSON 文件", "*.json"), ("所有文件", "*.*")],
        )
        if selected:
            self.result_var.set(selected)
            if not self.output_var.get().strip():
                self.output_var.set(str(Path(selected).parent))

    def _choose_output(self) -> None:
        selected = filedialog.askdirectory(title="选择文字输出目录")
        if selected:
            self.output_var.set(selected)

    def _append_log(self, message: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", message.rstrip() + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _start(self) -> None:
        if self.running:
            return
        video = Path(self.video_var.get().strip())
        result = Path(self.result_var.get().strip())
        if not video.is_file():
            messagebox.showerror("无法开始", "请选择有效的课堂视频。")
            return
        if not result.is_file():
            messagebox.showerror("无法开始", "请选择有效的 PPT result.json。")
            return
        output_text = self.output_var.get().strip()
        output = Path(output_text) if output_text else result.parent
        try:
            config_path = Path(self.config_var.get().strip())
            config = (
                TranscriptionConfig.from_file(config_path)
                if config_path.is_file()
                else TranscriptionConfig()
            )
            config = replace(
                config,
                model=self.model_var.get().strip(),
                hotwords=self.hotwords_var.get().strip() or None,
            )
            config.validate()
        except ValueError as exc:
            messagebox.showerror("配置错误", str(exc))
            return

        self.running = True
        self.progress_var.set(0)
        self.status_var.set("正在准备")
        self.start_button.configure(state="disabled")
        self.open_markdown_button.configure(state="disabled")
        self.open_output_button.configure(state="disabled")
        for item in self.result_tree.get_children():
            self.result_tree.delete(item)
        self._append_log(
            "首次使用某个模型时需要下载模型文件，请保持网络可用。"
        )

        def progress(message: str, value: float | None) -> None:
            self.events.put(("progress", message, value))

        def work() -> None:
            try:
                payload = transcribe_video_pages(
                    video,
                    result,
                    config=config,
                    output_dir=output,
                    progress_callback=progress,
                )
                self.events.put(("complete", payload))
            except Exception:
                self.events.put(("error", traceback.format_exc()))

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def _poll_events(self) -> None:
        try:
            while True:
                event = self.events.get_nowait()
                kind = event[0]
                if kind == "progress":
                    message = str(event[1])
                    value = event[2]
                    self.status_var.set(message)
                    if value is not None:
                        self.progress_var.set(float(value) * 100)
                    self._append_log(message)
                elif kind == "complete":
                    self._complete(event[1])
                elif kind == "error":
                    self._fail(str(event[1]))
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def _complete(self, raw_payload: object) -> None:
        payload = dict(raw_payload)  # type: ignore[arg-type]
        artifacts = payload["artifacts"]
        self.last_json_path = Path(artifacts["transcript_json"])
        self.last_markdown_path = Path(artifacts["page_transcript_markdown"])
        for page in payload["pages"]:
            preview = str(page["speech_text"]).replace("\n", " ")
            if len(preview) > 100:
                preview = preview[:100] + "…"
            self.result_tree.insert(
                "",
                "end",
                values=(
                    page["page_id"],
                    (
                        f"{format_timestamp(page['start_sec'])} ～ "
                        f"{format_timestamp(page['end_sec'])}"
                    ),
                    preview or "（没有识别到讲话）",
                ),
            )
        self.running = False
        self.progress_var.set(100)
        self.status_var.set(
            f"完成：识别 {payload['transcription']['utterance_count']} 段讲话"
        )
        self.start_button.configure(state="normal")
        self.open_markdown_button.configure(state="normal")
        self.open_output_button.configure(state="normal")
        self._append_log("处理完成，未保留任何音频文件。")

    def _fail(self, details: str) -> None:
        self.running = False
        self.start_button.configure(state="normal")
        self.status_var.set("处理失败")
        self._append_log(details)
        messagebox.showerror(
            "语音识别失败",
            "处理没有完成，请查看界面中的错误信息。",
        )

    def _open_markdown(self) -> None:
        if self.last_markdown_path and self.last_markdown_path.is_file():
            open_external(self.last_markdown_path)

    def _open_output(self) -> None:
        if self.last_json_path:
            open_external(self.last_json_path.parent)


def main() -> int:
    root = tk.Tk()
    TranscriptionApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
