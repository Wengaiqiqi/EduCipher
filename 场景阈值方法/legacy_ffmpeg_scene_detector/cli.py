from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import DetectorConfig
from .ffmpeg_io import FFmpegError
from .pipeline import VideoPageDetector


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="legacy-ffmpeg-scene-detector",
        description="归档版：基于 FFmpeg 场景阈值的 PPT 换页检测器",
    )
    parser.add_argument("video", type=Path, nargs="?", help="输入视频")
    parser.add_argument(
        "--gui",
        action="store_true",
        help="启动旧版图形界面",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config" / "default.json",
        help="旧版 JSON 配置",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("legacy_output"),
        help="输出目录",
    )
    parser.add_argument("--video-id", help="自定义视频 ID")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.gui:
        from .gui import main as gui_main

        return gui_main()
    if args.video is None:
        parser.error("请提供视频路径，或使用 --gui 启动图形界面")
    try:
        config = DetectorConfig.from_file(args.config)
        result = VideoPageDetector(config).run(
            args.video,
            output_root=args.output,
            video_id=args.video_id,
        )
    except (FFmpegError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    result_path = args.output / result["video_id"] / "result.json"
    print(
        json.dumps(
            {
                "status": "ok",
                "method": "legacy_ffmpeg_scene_threshold",
                "result_path": result_path.as_posix(),
                "page_count": len(result["pages"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0
