from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import DetectorConfig
from .evaluation import evaluate_results
from .ffmpeg_io import FFmpegError
from .llm_evaluation import LLMEvaluationConfig, evaluate_transcript
from .pipeline import VideoPageDetector
from .transcription import TranscriptionConfig, transcribe_video_pages


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video-page-detector",
        description="课堂录屏 PPT 换页检测器",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    detect = subparsers.add_parser("detect", help="检测视频中的 PPT 换页")
    detect.add_argument("video", type=Path, help="课堂录屏视频路径")
    detect.add_argument(
        "--config",
        type=Path,
        default=Path("config/default.json"),
        help="JSON 配置文件路径",
    )
    detect.add_argument(
        "--output",
        type=Path,
        default=Path("output"),
        help="输出根目录",
    )
    detect.add_argument("--video-id", help="输出中的视频 ID，默认使用文件名")

    evaluate = subparsers.add_parser("evaluate", help="计算 PRD 验收指标")
    evaluate.add_argument("predicted", type=Path, help="系统输出 result.json")
    evaluate.add_argument("ground_truth", type=Path, help="人工标注 JSON")
    evaluate.add_argument(
        "--tolerance",
        type=float,
        default=2.0,
        help="换页匹配容差（秒），默认 2",
    )

    config = subparsers.add_parser("print-config", help="打印内置默认配置")
    config.add_argument(
        "--compact",
        action="store_true",
        help="输出单行 JSON",
    )
    subparsers.add_parser("gui", help="启动桌面图形界面")
    subparsers.add_parser("transcribe-gui", help="启动逐页语音转文字界面")
    subparsers.add_parser("llm-evaluation-gui", help="启动PPT讲话关联度评估界面")

    transcribe = subparsers.add_parser(
        "transcribe",
        help="将视频中的全部讲话按 PPT 页面整理为文字",
    )
    transcribe.add_argument("video", type=Path, help="课堂录屏视频路径")
    transcribe.add_argument("result", type=Path, help="PPT 检测 result.json")
    transcribe.add_argument(
        "--config",
        type=Path,
        default=Path("config/transcription.json"),
        help="语音识别配置文件",
    )
    transcribe.add_argument(
        "--output",
        type=Path,
        help="文字输出目录，默认与 result.json 相同",
    )
    transcribe.add_argument(
        "--model",
        help="临时覆盖模型名称，例如 tiny、base、small、medium",
    )
    transcribe.add_argument(
        "--engine",
        choices=("faster-whisper", "mimo-cloud"),
        help="临时覆盖语音引擎；小米云端从MIMO_API_KEY读取密钥",
    )
    transcribe.add_argument(
        "--beam-size",
        type=int,
        help="临时覆盖搜索宽度；CPU 推荐 1",
    )

    llm_evaluate = subparsers.add_parser(
        "evaluate-llm",
        help="使用OpenAI兼容多模态LLM评估PPT与讲话关联度",
    )
    llm_evaluate.add_argument(
        "transcript",
        type=Path,
        help="逐页语音转文字生成的 transcript.json",
    )
    llm_evaluate.add_argument(
        "--config",
        type=Path,
        default=Path("config/llm_evaluation.json"),
        help="LLM服务配置文件",
    )
    llm_evaluate.add_argument(
        "--output",
        type=Path,
        help="评估输出目录，默认在转写目录下创建 llm_evaluation",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "detect":
            config = DetectorConfig.from_file(args.config)
            result = VideoPageDetector(config).run(
                args.video,
                output_root=args.output,
                video_id=args.video_id,
            )
            result_path = Path(args.output) / result["video_id"] / "result.json"
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "result_path": result_path.as_posix(),
                        "page_count": len(result["pages"]),
                        "review_page_count": sum(
                            page["confidence"] != "high"
                            for page in result["pages"]
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        elif args.command == "evaluate":
            metrics = evaluate_results(
                args.predicted,
                args.ground_truth,
                tolerance_sec=args.tolerance,
            )
            print(json.dumps(metrics, ensure_ascii=False, indent=2))
        elif args.command == "print-config":
            config = DetectorConfig()
            print(
                json.dumps(
                    config.to_dict(),
                    ensure_ascii=False,
                    indent=None if args.compact else 2,
                )
            )
        elif args.command == "transcribe":
            config = TranscriptionConfig.from_file(args.config)
            if args.engine or args.model or args.beam_size:
                config = TranscriptionConfig.from_mapping(
                    {
                        **config.to_dict(),
                        **({"engine": args.engine} if args.engine else {}),
                        **({"model": args.model} if args.model else {}),
                        **(
                            {"beam_size": args.beam_size}
                            if args.beam_size
                            else {}
                        ),
                    }
                )
            result = transcribe_video_pages(
                args.video,
                args.result,
                config=config,
                output_dir=args.output,
            )
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "page_count": len(result["pages"]),
                        "utterance_count": result["transcription"][
                            "utterance_count"
                        ],
                        **result["artifacts"],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        elif args.command == "evaluate-llm":
            config = LLMEvaluationConfig.from_file(args.config)
            result = evaluate_transcript(
                args.transcript,
                config=config,
                output_dir=args.output,
            )
            print(
                json.dumps(
                    {
                        "status": (
                            "ok" if result["summary"]["complete"] else "partial"
                        ),
                        **result["summary"],
                        **result["artifacts"],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        elif args.command == "transcribe-gui":
            from .transcription_gui import main as transcription_gui_main

            return transcription_gui_main()
        elif args.command == "llm-evaluation-gui":
            from .llm_evaluation_gui import main as llm_evaluation_gui_main

            return llm_evaluation_gui_main()
        else:
            from .gui import main as gui_main

            return gui_main()
    except (FFmpegError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
