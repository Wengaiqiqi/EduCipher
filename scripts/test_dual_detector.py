from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from video_page_detector.dual_detector import (
    SceneHint,
    analyze_reuse,
    filter_short_scene_intervals,
    load_temporal_result,
    stream_scene_hints,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="在真实视频上测试场景提示与时序页面的复用率"
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("temporal_result", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--tolerance", type=float, default=2.0)
    parser.add_argument("--min-interval", type=float, default=5.0)
    args = parser.parse_args()

    temporal = load_temporal_result(args.temporal_result)
    crop = temporal.get("analysis", {}).get("screen_crop_ratios", {})
    crop_ratios = (
        float(crop.get("left", 0.1)),
        float(crop.get("top", 0.02)),
        float(crop.get("right", 0.1)),
        float(crop.get("bottom", 0.1)),
    )
    started = time.perf_counter()

    def report(hint: SceneHint) -> None:
        print(
            json.dumps(
                {
                    "scene_sec": round(hint.timestamp_sec, 3),
                    "wall_sec": round(hint.emitted_after_sec, 3),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    hints = stream_scene_hints(
        args.video,
        threshold=args.threshold,
        crop_ratios=crop_ratios,
        callback=report,
    )
    filtered_hints = filter_short_scene_intervals(
        hints,
        min_interval_sec=args.min_interval,
    )
    report_payload = analyze_reuse(
        temporal["pages"],
        filtered_hints,
        video_duration_sec=float(temporal["video_duration_sec"]),
        tolerance_sec=args.tolerance,
    )
    report_payload.update(
        {
            "video_path": args.video.resolve().as_posix(),
            "temporal_result_path": (
                args.temporal_result.resolve().as_posix()
            ),
            "scene_threshold": args.threshold,
            "boundary_tolerance_sec": args.tolerance,
            "minimum_scene_interval_sec": args.min_interval,
            "raw_scene_hint_count": len(hints),
            "scene_scan_duration_sec": round(
                time.perf_counter() - started,
                3,
            ),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: report_payload[key]
                for key in (
                    "scene_scan_duration_sec",
                    "scene_hint_count",
                    "temporal_boundary_count",
                    "matched_boundary_count",
                    "boundary_match_rate",
                    "reusable_page_count",
                    "reprocess_page_count",
                    "page_reuse_rate",
                    "first_scene_hint_emitted_after_sec",
                    "last_scene_hint_emitted_after_sec",
                )
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
