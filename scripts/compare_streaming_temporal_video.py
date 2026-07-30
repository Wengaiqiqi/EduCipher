from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from video_page_detector.config import DetectorConfig  # noqa: E402
from video_page_detector.pipeline import VideoPageDetector  # noqa: E402
from video_page_detector.streaming_pipeline import (  # noqa: E402
    StreamingVideoPageDetector,
)


def _sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--label")
    args = parser.parse_args()

    video = args.video.resolve()
    label = args.label or video.stem
    output_root = (
        PROJECT_ROOT / "diagnostics" / f"streaming_temporal_comparison_{label}"
    )
    config = DetectorConfig.from_file(
        PROJECT_ROOT / "config" / "default.json"
    )

    batch_started = time.perf_counter()
    batch = VideoPageDetector(config).run(
        video,
        output_root=output_root,
        video_id=f"{label}_batch",
        progress_callback=lambda message, progress: print(
            json.dumps(
                {
                    "mode": "batch",
                    "elapsed_sec": round(
                        time.perf_counter() - batch_started,
                        3,
                    ),
                    "message": message,
                    "progress": progress,
                },
                ensure_ascii=False,
            ),
            flush=True,
        ),
    )
    batch_total = round(time.perf_counter() - batch_started, 3)

    streaming_started = time.perf_counter()
    page_events: list[dict[str, float | int]] = []

    def page_ready(page: dict, completed: int, total: int) -> None:
        event = {
            "page_id": int(page["page_id"]),
            "ready_after_sec": round(
                time.perf_counter() - streaming_started,
                3,
            ),
            "completed": completed,
            "known_total": total,
        }
        page_events.append(event)
        print(json.dumps(event, ensure_ascii=False), flush=True)

    streaming = StreamingVideoPageDetector(config).run(
        video,
        output_root=output_root,
        video_id=f"{label}_streaming",
        progress_callback=lambda message, progress: print(
            json.dumps(
                {
                    "mode": "streaming",
                    "elapsed_sec": round(
                        time.perf_counter() - streaming_started,
                        3,
                    ),
                    "message": message,
                    "progress": progress,
                },
                ensure_ascii=False,
            ),
            flush=True,
        ),
        page_ready_callback=page_ready,
    )
    streaming_total = round(time.perf_counter() - streaming_started, 3)

    paired = list(zip(batch["pages"], streaming["pages"]))
    boundary_errors = [
        max(
            abs(float(previous["start_sec"]) - float(current["start_sec"])),
            abs(float(previous["end_sec"]) - float(current["end_sec"])),
        )
        for previous, current in paired
    ]
    screenshots_equal = (
        len(batch["pages"]) == len(streaming["pages"])
        and all(
            _sha256(previous["screenshot_path"])
            == _sha256(current["screenshot_path"])
            for previous, current in paired
        )
    )
    summary = {
        "video_path": video.as_posix(),
        "video_duration_sec": streaming["video_duration_sec"],
        "batch_total_sec": batch_total,
        "streaming_total_sec": streaming_total,
        "batch_page_count": len(batch["pages"]),
        "streaming_page_count": len(streaming["pages"]),
        "first_page_ready_after_sec": (
            page_events[0]["ready_after_sec"] if page_events else None
        ),
        "max_boundary_error_sec": max(boundary_errors, default=None),
        "screenshots_equal": screenshots_equal,
        "intro_pages_ending_within_30_sec": sum(
            float(page["end_sec"]) <= 30.0
            for page in streaming["pages"]
        ),
        "first_pages": [
            {
                "page_id": page["page_id"],
                "start_sec": page["start_sec"],
                "end_sec": page["end_sec"],
                "representative_sec": page["representative_sec"],
            }
            for page in streaming["pages"][:3]
        ],
        "page_events": page_events,
    }
    summary_path = output_root / "comparison.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
