from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image

from .hashing import grid_hashes, hamming_distance, phash


@dataclass(frozen=True)
class TemporalFeature:
    timestamp_sec: float
    block_hashes: np.ndarray
    global_hash: np.ndarray
    information_score: float
    content_line_hashes: tuple[np.ndarray, ...] = ()


@dataclass(frozen=True)
class TemporalSegment:
    start_index: int
    end_index: int
    representative_index: int
    change_ratio: float
    confidence: str


def make_temporal_feature(
    frame: np.ndarray,
    *,
    timestamp_sec: float,
    rows: int,
    columns: int,
) -> TemporalFeature:
    gray = (
        frame.astype(np.float32).mean(axis=2)
        if frame.ndim == 3
        else frame.astype(np.float32)
    )
    horizontal_edges = np.abs(np.diff(gray, axis=1)).mean()
    vertical_edges = np.abs(np.diff(gray, axis=0)).mean()
    information_score = float((horizontal_edges + vertical_edges) / 510.0)
    return TemporalFeature(
        timestamp_sec=timestamp_sec,
        block_hashes=grid_hashes(frame, rows=rows, columns=columns),
        global_hash=phash(frame),
        information_score=information_score,
        content_line_hashes=extract_content_line_hashes(frame),
    )


def extract_content_line_hashes(frame: np.ndarray) -> tuple[np.ndarray, ...]:
    """Build a position-independent signature from visible text/content rows."""
    gray = np.asarray(
        Image.fromarray(frame.astype(np.uint8, copy=False))
        .convert("L")
        .resize((320, 180), Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )
    height, width = gray.shape
    ink = gray < 185

    # Ignore the fixed course banner/footer and the usual presenter overlay.
    # Those regions repeat on every slide and must not inflate similarity.
    ink[: int(height * 0.10), :] = False
    ink[int(height * 0.94) :, :] = False
    ink[int(height * 0.72) :, int(width * 0.78) :] = False

    active_rows = np.mean(ink, axis=1) >= 0.008
    bands: list[tuple[int, int]] = []
    start: int | None = None
    quiet_rows = 0
    for row, active in enumerate(active_rows):
        if active:
            if start is None:
                start = row
            quiet_rows = 0
        elif start is not None:
            quiet_rows += 1
            if quiet_rows > 2:
                bands.append((start, row - quiet_rows + 1))
                start = None
                quiet_rows = 0
    if start is not None:
        bands.append((start, height))

    hashes: list[np.ndarray] = []
    for y0, y1 in bands:
        _, columns = np.where(ink[y0:y1])
        if len(columns) < 30 or y1 - y0 < 2:
            continue
        x0 = int(columns.min())
        x1 = int(columns.max()) + 1
        if x1 - x0 < 6:
            continue
        hashes.append(phash(gray[y0:y1, x0:x1]))
    return tuple(hashes)


def content_line_similarity(
    first: Sequence[np.ndarray],
    second: Sequence[np.ndarray],
    *,
    maximum_hash_distance: int = 14,
) -> float:
    """Compare line content while ignoring line position and vertical spacing."""
    if not first or not second:
        return 0.0
    remaining = list(range(len(second)))
    match_count = 0
    for first_hash in first:
        if not remaining:
            break
        best = min(
            remaining,
            key=lambda index: hamming_distance(first_hash, second[index]),
        )
        if hamming_distance(first_hash, second[best]) <= maximum_hash_distance:
            match_count += 1
            remaining.remove(best)
    return match_count / max(len(first), len(second))


def consensus_hash(hashes: np.ndarray) -> np.ndarray:
    if hashes.ndim < 2 or hashes.shape[0] == 0:
        raise ValueError("at least one hash is required")
    return np.mean(hashes.astype(np.float32), axis=0) >= 0.5


def block_distances(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    if first.shape != second.shape:
        raise ValueError("block hash arrays must have identical shapes")
    return np.count_nonzero(first != second, axis=1)


def feature_distance(first: TemporalFeature, second: TemporalFeature) -> float:
    block_distance = float(
        np.mean(block_distances(first.block_hashes, second.block_hashes)) / 64.0
    )
    global_distance = hamming_distance(first.global_hash, second.global_hash) / 64.0
    return 0.7 * block_distance + 0.3 * global_distance


def find_state_crossover(
    samples: Sequence[TemporalFeature],
    *,
    before_state: TemporalFeature,
    after_state: TemporalFeature,
    persistence: int = 3,
) -> int | None:
    if persistence < 1:
        raise ValueError("persistence must be positive")
    closer_to_after = [
        feature_distance(sample, after_state)
        < feature_distance(sample, before_state)
        for sample in samples
    ]
    for index in range(0, len(closer_to_after) - persistence + 1):
        if all(closer_to_after[index : index + persistence]):
            return index
    return None


def _consensus_blocks(features: Sequence[TemporalFeature]) -> np.ndarray:
    return consensus_hash(
        np.stack([feature.block_hashes for feature in features])
    )


def _candidate_change_ratio(
    before: Sequence[TemporalFeature],
    after: Sequence[TemporalFeature],
    *,
    block_distance_threshold: int,
    stability_distance: int,
) -> tuple[float, float]:
    before_consensus = _consensus_blocks(before)
    after_consensus = _consensus_blocks(after)
    changed_distances = block_distances(before_consensus, after_consensus)
    after_stack = np.stack([feature.block_hashes for feature in after])
    stability = np.mean(
        np.count_nonzero(after_stack != after_consensus[None, :, :], axis=2),
        axis=0,
    )
    stable_changes = (
        (changed_distances >= block_distance_threshold)
        & (stability <= stability_distance)
    )
    change_ratio = float(np.mean(stable_changes))
    before_global = consensus_hash(
        np.stack([feature.global_hash for feature in before])
    )
    after_global = consensus_hash(
        np.stack([feature.global_hash for feature in after])
    )
    global_distance = float(hamming_distance(before_global, after_global))
    return change_ratio, global_distance


def find_temporal_segments(
    features: Sequence[TemporalFeature],
    *,
    confirmation_samples: int = 3,
    changed_block_ratio: float = 0.34,
    block_distance_threshold: int = 10,
    stability_distance: int = 12,
    minimum_segment_samples: int = 3,
    same_content_similarity: float = 0.80,
) -> list[TemporalSegment]:
    if not features:
        return []
    if confirmation_samples < 2:
        raise ValueError("confirmation_samples must be at least 2")
    boundaries: list[tuple[int, float, float]] = [(0, 1.0, 64.0)]
    index = minimum_segment_samples
    count = len(features)
    while index + confirmation_samples <= count:
        if index - boundaries[-1][0] < minimum_segment_samples:
            index += 1
            continue
        before = features[
            max(boundaries[-1][0], index - confirmation_samples) : index
        ]
        after = features[index : index + confirmation_samples]
        ratio, global_distance = _candidate_change_ratio(
            before,
            after,
            block_distance_threshold=block_distance_threshold,
            stability_distance=stability_distance,
        )
        # A shared slide template can keep the global pHash almost unchanged
        # even when most body blocks are replaced.  Persistence and the
        # changed-block ratio already reject localized/transient motion, so
        # global distance remains diagnostic rather than a hard gate.
        if ratio >= changed_block_ratio:
            confirmed_index = min(
                count - 1,
                index + confirmation_samples // 2,
            )
            boundaries.append((confirmed_index, ratio, global_distance))
            index = confirmed_index + confirmation_samples
        else:
            index += 1

    segments: list[TemporalSegment] = []
    for boundary_index, (start, ratio, _) in enumerate(boundaries):
        end = (
            boundaries[boundary_index + 1][0]
            if boundary_index + 1 < len(boundaries)
            else count
        )
        if end <= start:
            continue
        representative = choose_representative_index(features, start, end)
        confidence = "high" if ratio >= 0.6 else "low"
        segments.append(
            TemporalSegment(
                start_index=start,
                end_index=end,
                representative_index=representative,
                change_ratio=ratio,
                confidence=confidence,
            )
        )
    return merge_same_content_segments(
        features,
        segments,
        similarity_threshold=same_content_similarity,
    )


def merge_same_content_segments(
    features: Sequence[TemporalFeature],
    segments: Sequence[TemporalSegment],
    *,
    similarity_threshold: float = 0.80,
) -> list[TemporalSegment]:
    """Merge animation states that contain the same lines in new positions."""
    merged: list[TemporalSegment] = []
    for segment in segments:
        if merged:
            previous = merged[-1]
            previous_feature = features[previous.representative_index]
            current_feature = features[segment.representative_index]
            similarity = content_line_similarity(
                previous_feature.content_line_hashes,
                current_feature.content_line_hashes,
            )
            if similarity >= similarity_threshold:
                combined_end = segment.end_index
                merged[-1] = TemporalSegment(
                    start_index=previous.start_index,
                    end_index=combined_end,
                    representative_index=choose_representative_index(
                        features,
                        previous.start_index,
                        combined_end,
                    ),
                    change_ratio=previous.change_ratio,
                    confidence=previous.confidence,
                )
                continue
        merged.append(segment)
    return merged


def choose_representative_index(
    features: Sequence[TemporalFeature],
    start: int,
    end: int,
) -> int:
    if not 0 <= start < end <= len(features):
        raise ValueError("invalid segment bounds")
    segment = features[start:end]
    medoid_blocks = _consensus_blocks(segment)
    first_blocks = _consensus_blocks(segment[: min(2, len(segment))])
    membership = np.array(
        [
            float(np.mean(block_distances(feature.block_hashes, medoid_blocks)))
            for feature in segment
        ]
    )
    gains = np.array(
        [
            float(np.mean(block_distances(feature.block_hashes, first_blocks)))
            for feature in segment
        ]
    )
    information = np.array(
        [feature.information_score for feature in segment],
        dtype=np.float64,
    )
    if len(segment) == 1:
        return start
    membership_limit = float(np.quantile(membership, 0.75))
    gain_scale = max(1.0, float(np.max(gains)))
    info_scale = max(1e-6, float(np.max(information)))
    progress = np.linspace(0.0, 1.0, len(segment))
    score = gains / gain_scale + 0.2 * information / info_scale + 0.15 * progress
    score[membership > membership_limit] -= 1.0
    return start + int(np.argmax(score))
