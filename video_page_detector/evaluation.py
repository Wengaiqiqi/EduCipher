from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_result(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid result JSON: {path}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("pages"), list):
        raise ValueError(f"Result JSON must contain a pages list: {path}")
    return payload


def evaluate_results(
    predicted_path: str | Path,
    ground_truth_path: str | Path,
    *,
    tolerance_sec: float = 2.0,
) -> dict[str, Any]:
    if tolerance_sec <= 0:
        raise ValueError("tolerance_sec must be positive")
    predicted = _load_result(predicted_path)
    ground_truth = _load_result(ground_truth_path)
    predicted_times = [
        float(page["start_sec"])
        for page in predicted["pages"]
        if float(page["start_sec"]) > 0
    ]
    truth_times = [
        float(page["start_sec"])
        for page in ground_truth["pages"]
        if float(page["start_sec"]) > 0
    ]

    unmatched = set(range(len(predicted_times)))
    errors: list[float] = []
    matches: list[dict[str, float]] = []
    missed: list[float] = []
    for truth_time in truth_times:
        candidates = sorted(
            (
                (abs(predicted_times[index] - truth_time), index)
                for index in unmatched
            ),
            key=lambda item: item[0],
        )
        if candidates and candidates[0][0] <= tolerance_sec:
            error, index = candidates[0]
            unmatched.remove(index)
            errors.append(error)
            matches.append(
                {
                    "ground_truth_sec": truth_time,
                    "predicted_sec": predicted_times[index],
                    "absolute_error_sec": round(error, 3),
                }
            )
        else:
            missed.append(truth_time)

    false_positives = [predicted_times[index] for index in sorted(unmatched)]
    recall = len(matches) / len(truth_times) if truth_times else 1.0
    false_positive_rate = (
        len(false_positives) / len(predicted_times) if predicted_times else 0.0
    )
    max_error = max(errors) if errors else None
    mean_error = sum(errors) / len(errors) if errors else None
    processing_duration = predicted.get("processing_duration_sec")
    video_duration = predicted.get("video_duration_sec")
    performance_pass = None
    if processing_duration is not None and video_duration is not None:
        performance_pass = float(processing_duration) <= (
            float(video_duration) / 3600.0 * 300.0
        )

    return {
        "tolerance_sec": tolerance_sec,
        "ground_truth_change_count": len(truth_times),
        "predicted_change_count": len(predicted_times),
        "matched_count": len(matches),
        "recall": round(recall, 4),
        "false_positive_rate": round(false_positive_rate, 4),
        "mean_timestamp_error_sec": (
            round(mean_error, 3) if mean_error is not None else None
        ),
        "max_timestamp_error_sec": (
            round(max_error, 3) if max_error is not None else None
        ),
        "matches": matches,
        "missed_ground_truth_sec": missed,
        "false_positive_sec": false_positives,
        "acceptance": {
            "recall_at_least_90_percent": recall >= 0.9,
            "false_positive_rate_at_most_10_percent": false_positive_rate <= 0.1,
            "timestamp_error_at_most_2_sec": (
                bool(errors)
                and len(errors) == len(truth_times)
                and max(errors) <= 2.0
            )
            if truth_times
            else True,
            "processing_time_within_5_min_per_hour": performance_pass,
        },
    }
