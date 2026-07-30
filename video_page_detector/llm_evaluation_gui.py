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

from .llm_evaluation import LLMEvaluationConfig, evaluate_transcript


def open_external(path: Path) -> None:
    target = str(path.resolve())
    if sys.platform == "win32":
        os.startfile(target)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", target])
    else:
        subprocess.Popen(["xdg-open", target])


class LLMEvaluationApp:
    BACKGROUND = "#F4F7FB"
    CARD = "#FFFFFF"
    PRIMARY = "#2563EB"
    TEXT = "#172033"
    MUTED = "#64748B"

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("PPT与讲话关联度评估")
        self.root.geometry("1080x840")
        self.root.minsize(900, 700)
        self.root.configure(background=self.BACKGROUND)

        self.events: queue.Queue[tuple[object, ...]] = queue.Queue()
        self.running = False
        self.last_report_path: Path | None = None
        self.last_result_path: Path | None = None

        self.transcript_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.config_var = tk.StringVar(
            value=str(Path.cwd() / "config" / "llm_evaluation.json")
        )
        self.base_url_var = tk.StringVar(value="https://api.openai.com/v1")
        self.model_var = tk.StringVar(value="gpt-4o-mini")
        self.api_key_var = tk.StringVar()
        self.concurrency_var = tk.StringVar(value="5")
        self.include_evidence_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="请选择 transcript.json")
        self.progress_var = tk.DoubleVar(value=0)

        self._configure_styles()
        self._build_interface()
        self._load_default_config()
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
        style.configure("Treeview", rowheight=32)
        style.configure(
            "Treeview.Heading",
            font=("Microsoft YaHei UI", 9, "bold"),
        )

    def _build_interface(self) -> None:
        main = ttk.Frame(self.root, style="App.TFrame", padding=(24, 18))
        main.pack(fill="both", expand=True)
        ttk.Label(
            main,
            text="PPT与讲话关联度评估",
            style="Title.TLabel",
        ).pack(anchor="w")
        ttk.Label(
            main,
            text=(
                "每页独立发送PPT截图和讲话文字，默认并发5；"
                "默认仅返回分数和理由，可按需开启详细证据"
            ),
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(4, 14))

        input_card = ttk.Frame(main, style="Card.TFrame", padding=16)
        input_card.pack(fill="x")
        input_card.columnconfigure(1, weight=1)
        ttk.Label(
            input_card,
            text="服务与输入",
            style="CardTitle.TLabel",
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
        self._path_row(
            input_card,
            1,
            "转写结果",
            self.transcript_var,
            self._choose_transcript,
            "选择 transcript.json",
        )
        self._path_row(
            input_card,
            2,
            "输出目录",
            self.output_var,
            self._choose_output,
            "选择目录",
        )
        self._entry_row(
            input_card,
            3,
            "Base URL",
            self.base_url_var,
            "例如：https://服务地址/v1",
        )
        self._entry_row(
            input_card,
            4,
            "模型名称",
            self.model_var,
            "必须是支持图片输入的多模态模型",
        )

        ttk.Label(
            input_card,
            text="API密钥",
            style="CardText.TLabel",
        ).grid(row=5, column=0, sticky="w", padx=(0, 12), pady=5)
        ttk.Entry(
            input_card,
            textvariable=self.api_key_var,
            show="•",
        ).grid(row=5, column=1, sticky="ew", pady=5)
        ttk.Label(
            input_card,
            text="仅保存在当前界面内存，不写入磁盘",
            style="Help.TLabel",
        ).grid(row=5, column=2, sticky="w", padx=(12, 0), pady=5)

        ttk.Label(
            input_card,
            text="并发数量",
            style="CardText.TLabel",
        ).grid(row=6, column=0, sticky="w", padx=(0, 12), pady=5)
        ttk.Spinbox(
            input_card,
            from_=1,
            to=10,
            textvariable=self.concurrency_var,
            width=8,
        ).grid(row=6, column=1, sticky="w", pady=5)
        ttk.Label(
            input_card,
            text="默认5，允许1～10；遇到限流时调小",
            style="Help.TLabel",
        ).grid(row=6, column=2, sticky="w", padx=(12, 0), pady=5)

        ttk.Checkbutton(
            input_card,
            text="返回详细对应证据（增加Token消耗）",
            variable=self.include_evidence_var,
        ).grid(row=7, column=0, columnspan=3, sticky="w", pady=(8, 0))

        action_row = ttk.Frame(input_card, style="Card.TFrame")
        action_row.grid(row=8, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        self.start_button = ttk.Button(
            action_row,
            text="开始关联度评估",
            style="Primary.TButton",
            command=self._start,
        )
        self.start_button.pack(side="left")
        self.open_report_button = ttk.Button(
            action_row,
            text="打开评估报告",
            style="Secondary.TButton",
            command=self._open_report,
            state="disabled",
        )
        self.open_report_button.pack(side="left", padx=(10, 0))
        self.open_output_button = ttk.Button(
            action_row,
            text="打开输出目录",
            style="Secondary.TButton",
            command=self._open_output_dir,
            state="disabled",
        )
        self.open_output_button.pack(side="left", padx=(8, 0))

        status_card = ttk.Frame(main, style="Card.TFrame", padding=16)
        status_card.pack(fill="x", pady=(14, 0))
        header = ttk.Frame(status_card, style="Card.TFrame")
        header.pack(fill="x")
        ttk.Label(
            header,
            text="处理状态",
            style="CardTitle.TLabel",
        ).pack(side="left")
        ttk.Label(
            header,
            textvariable=self.status_var,
            style="Help.TLabel",
        ).pack(side="right")
        ttk.Progressbar(
            status_card,
            variable=self.progress_var,
            maximum=100,
        ).pack(fill="x", pady=(12, 10))
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
            text="页面评分",
            style="CardTitle.TLabel",
        ).pack(anchor="w", pady=(0, 8))
        columns = (
            "page",
            "status",
            "relevance",
            "coverage",
            "evidence",
            "score",
            "level",
        )
        self.tree = ttk.Treeview(
            result_card,
            columns=columns,
            show="headings",
            height=9,
        )
        labels = {
            "page": "页码",
            "status": "状态",
            "relevance": "讲话相关度",
            "coverage": "PPT覆盖度",
            "evidence": "证据一致性",
            "score": "页面分数",
            "level": "等级",
        }
        widths = {
            "page": 65,
            "status": 95,
            "relevance": 110,
            "coverage": 100,
            "evidence": 100,
            "score": 90,
            "level": 110,
        }
        for column in columns:
            self.tree.heading(column, text=labels[column])
            self.tree.column(
                column,
                width=widths[column],
                anchor="center",
            )
        scrollbar = ttk.Scrollbar(
            result_card,
            orient="vertical",
            command=self.tree.yview,
        )
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side="left", fill="both", expand=True)
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
        ttk.Label(parent, text=label, style="CardText.TLabel").grid(
            row=row,
            column=0,
            sticky="w",
            padx=(0, 12),
            pady=5,
        )
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
        ).grid(row=row, column=2, padx=(12, 0), pady=5)

    def _entry_row(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.StringVar,
        help_text: str,
    ) -> None:
        ttk.Label(parent, text=label, style="CardText.TLabel").grid(
            row=row,
            column=0,
            sticky="w",
            padx=(0, 12),
            pady=5,
        )
        ttk.Entry(parent, textvariable=variable).grid(
            row=row,
            column=1,
            sticky="ew",
            pady=5,
        )
        ttk.Label(
            parent,
            text=help_text,
            style="Help.TLabel",
        ).grid(row=row, column=2, sticky="w", padx=(12, 0), pady=5)

    def _load_default_config(self) -> None:
        path = Path(self.config_var.get())
        if not path.is_file():
            return
        try:
            config = LLMEvaluationConfig.from_file(path)
        except ValueError:
            return
        self.base_url_var.set(config.base_url)
        self.model_var.set(config.model)
        self.concurrency_var.set(str(config.max_concurrency))
        self.include_evidence_var.set(config.include_evidence)

    def _choose_transcript(self) -> None:
        selected = filedialog.askopenfilename(
            title="选择 transcript.json",
            filetypes=[("JSON 文件", "*.json"), ("所有文件", "*.*")],
        )
        if selected:
            self.transcript_var.set(selected)
            self.output_var.set(str(Path(selected).parent / "llm_evaluation"))

    def _choose_output(self) -> None:
        selected = filedialog.askdirectory(title="选择评估输出目录")
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
        transcript = Path(self.transcript_var.get().strip())
        if not transcript.is_file():
            messagebox.showerror("无法开始", "请选择有效的 transcript.json。")
            return
        api_key = self.api_key_var.get().strip()
        if not api_key:
            messagebox.showerror("无法开始", "请输入本次使用的 API 密钥。")
            return
        try:
            concurrency = int(self.concurrency_var.get())
            config_path = Path(self.config_var.get())
            base = (
                LLMEvaluationConfig.from_file(config_path)
                if config_path.is_file()
                else LLMEvaluationConfig()
            )
            config = replace(
                base,
                base_url=self.base_url_var.get().strip(),
                model=self.model_var.get().strip(),
                max_concurrency=concurrency,
                include_evidence=bool(self.include_evidence_var.get()),
            )
            config.validate()
        except (TypeError, ValueError) as exc:
            messagebox.showerror("配置错误", str(exc))
            return
        output_text = self.output_var.get().strip()
        output = (
            Path(output_text)
            if output_text
            else transcript.parent / "llm_evaluation"
        )

        self.running = True
        self.progress_var.set(0)
        self.status_var.set("正在准备请求")
        self.start_button.configure(state="disabled")
        self.open_report_button.configure(state="disabled")
        self.open_output_button.configure(state="disabled")
        for item in self.tree.get_children():
            self.tree.delete(item)

        def progress(message: str, completed: int, total: int) -> None:
            self.events.put(("progress", message, completed, total))

        def work() -> None:
            try:
                result = evaluate_transcript(
                    transcript,
                    config=config,
                    output_dir=output,
                    api_key=api_key,
                    progress_callback=progress,
                )
                self.events.put(("complete", result))
            except Exception:
                self.events.put(("error", traceback.format_exc()))

        threading.Thread(target=work, daemon=True).start()

    def _poll_events(self) -> None:
        try:
            while True:
                event = self.events.get_nowait()
                if event[0] == "progress":
                    _, message, completed, total = event
                    self.status_var.set(f"{completed}/{total}：{message}")
                    self.progress_var.set(float(completed) / float(total) * 100)
                    self._append_log(str(message))
                elif event[0] == "complete":
                    self._complete(event[1])
                elif event[0] == "error":
                    self._fail(str(event[1]))
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def _complete(self, raw_result: object) -> None:
        result = dict(raw_result)  # type: ignore[arg-type]
        for page in result["pages"]:
            self.tree.insert(
                "",
                "end",
                values=(
                    page["page_id"],
                    page["status"],
                    page["speech_relevance"],
                    page["ppt_coverage"],
                    page["evidence_consistency"],
                    page["score"],
                    page["level"],
                ),
            )
        self.last_result_path = Path(result["artifacts"]["result_json"])
        self.last_report_path = Path(result["artifacts"]["report_markdown"])
        summary = result["summary"]
        self.running = False
        self.progress_var.set(100)
        self.status_var.set(
            (
                f"完成：严格总分 {summary['strict_overall_score']}，"
                f"纯关联平均分 {summary['association_average_score']}"
            )
        )
        self.start_button.configure(state="normal")
        self.open_report_button.configure(state="normal")
        self.open_output_button.configure(state="normal")
        self._append_log(self.status_var.get())
        self.api_key_var.set("")

    def _fail(self, details: str) -> None:
        self.running = False
        self.start_button.configure(state="normal")
        self.status_var.set("处理失败")
        self._append_log(details)
        self.api_key_var.set("")
        messagebox.showerror("评估失败", "请查看界面中的错误信息。")

    def _open_report(self) -> None:
        if self.last_report_path and self.last_report_path.is_file():
            open_external(self.last_report_path)

    def _open_output_dir(self) -> None:
        if self.last_result_path:
            open_external(self.last_result_path.parent)


def main() -> int:
    root = tk.Tk()
    LLMEvaluationApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
