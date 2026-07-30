from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from video_page_detector.config import DetectorConfig  # noqa: E402
from video_page_detector.streaming_pipeline import (  # noqa: E402
    StreamingVideoPageDetector,
)


def main() -> int:
    video = Path(r"E:\leeson\test_vedio\444.mp4")
    baseline_path = Path(
        r"E:\leeson\diagnostics\full_chain_444_mimo_streaming_v2"
        r"\444_mimo_streaming_v2\result.json"
    )
    output_root = PROJECT_ROOT / "diagnostics" / "streaming_temporal_444"
    video_id = "444_streaming_temporal"
    started = time.perf_counter()
    page_events: list[dict[str, float | int]] = []

    def elapsed() -> float:
        return round(time.perf_counter() - started, 3)

    def page_ready(page: dict, completed: int, total: int) -> None:
        event = {
            "page_id": int(page["page_id"]),
            "ready_after_sec": elapsed(),
            "completed": completed,
            "known_total": total,
        }
        page_events.append(event)
        print(json.dumps(event, ensure_ascii=False), flush=True)

    config = DetectorConfig.from_file(
        PROJECT_ROOT / "config" / "default.json"
    )
    result = StreamingVideoPageDetector(config).run(
        video,
        output_root=output_root,
        video_id=video_id,
        progress_callback=lambda message, progress: print(
            json.dumps(
                {
                    "elapsed_sec": elapsed(),
                    "message": message,
                    "progress": progress,
                },
                ensure_ascii=False,
            ),
            flush=True,
        ),
        page_ready_callback=page_ready,
    )
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    paired = list(zip(baseline["pages"], result["pages"]))
    boundary_errors = [
        {
            "page_id": int(current["page_id"]),
            "start_error_sec": round(
                abs(
                    float(current["start_sec"])
                    - float(previous["start_sec"])
                ),
                6,
            ),
            "end_error_sec": round(
                abs(
                    float(current["end_sec"])
                    - float(previous["end_sec"])
                ),
                6,
            ),
        }
        for previous, current in paired
    ]
    summary = {
        "total_sec": elapsed(),
        "baseline_detection_sec": baseline["processing_duration_sec"],
        "page_count": len(result["pages"]),
        "baseline_page_count": len(baseline["pages"]),
        "first_page_ready_after_sec": (
            page_events[0]["ready_after_sec"] if page_events else None
        ),
        "pages_ready_before_eof": sum(
            event["ready_after_sec"] < result["processing_duration_sec"]
            for event in page_events
        ),
        "max_start_error_sec": max(
            (item["start_error_sec"] for item in boundary_errors),
            default=None,
        ),
        "max_end_error_sec": max(
            (item["end_error_sec"] for item in boundary_errors),
            default=None,
        ),
        "page_events": page_events,
        "boundary_errors": boundary_errors,
    }
    run_dir = output_root / video_id
    (run_dir / "streaming_detection_performance.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
