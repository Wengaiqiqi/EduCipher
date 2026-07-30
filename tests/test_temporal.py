import unittest

import numpy as np

from video_page_detector.temporal import (
    IncrementalTemporalSegmenter,
    TemporalFeature,
    choose_representative_index,
    find_state_crossover,
    find_temporal_segments,
)


def _feature(timestamp: float, value: int, information: float = 0.1) -> TemporalFeature:
    block_hashes = np.zeros((12, 64), dtype=bool)
    global_hash = np.zeros(64, dtype=bool)
    if value:
        block_hashes[:value, :16] = True
        global_hash[: value * 4] = True
    return TemporalFeature(timestamp, block_hashes, global_hash, information)


class TemporalSegmentationTests(unittest.TestCase):
    @staticmethod
    def _segmenter() -> IncrementalTemporalSegmenter:
        return IncrementalTemporalSegmenter(
            confirmation_samples=3,
            changed_block_ratio=0.5,
            block_distance_threshold=10,
            stability_distance=12,
            minimum_segment_samples=3,
            same_content_similarity=0.8,
        )

    def test_persistent_change_creates_new_segment(self) -> None:
        features = [
            *[_feature(index * 2, 0) for index in range(6)],
            *[_feature((index + 6) * 2, 8) for index in range(6)],
        ]
        segments = find_temporal_segments(
            features,
            confirmation_samples=3,
            changed_block_ratio=0.5,
            minimum_segment_samples=3,
        )
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[1].start_index, 6)

    def test_same_global_template_with_changed_body_creates_segment(self) -> None:
        before = [_feature(index * 2, 0) for index in range(6)]
        after: list[TemporalFeature] = []
        for index in range(6):
            feature = _feature((index + 6) * 2, 8)
            after.append(
                TemporalFeature(
                    timestamp_sec=feature.timestamp_sec,
                    block_hashes=feature.block_hashes,
                    global_hash=np.zeros(64, dtype=bool),
                    information_score=feature.information_score,
                )
            )
        segments = find_temporal_segments(
            [*before, *after],
            confirmation_samples=3,
            changed_block_ratio=0.5,
            minimum_segment_samples=3,
        )
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[1].start_index, 6)

    def test_same_content_repositioning_is_merged(self) -> None:
        shared_lines = tuple(
            np.roll(np.arange(64) % 2 == 0, shift)
            for shift in range(4)
        )
        before = [
            TemporalFeature(
                timestamp_sec=index * 2,
                block_hashes=_feature(0, 0).block_hashes,
                global_hash=np.zeros(64, dtype=bool),
                information_score=0.1,
                content_line_hashes=shared_lines,
            )
            for index in range(6)
        ]
        after = [
            TemporalFeature(
                timestamp_sec=(index + 6) * 2,
                block_hashes=_feature(0, 8).block_hashes,
                global_hash=np.ones(64, dtype=bool),
                information_score=0.2,
                content_line_hashes=tuple(reversed(shared_lines)),
            )
            for index in range(6)
        ]
        segments = find_temporal_segments(
            [*before, *after],
            confirmation_samples=3,
            changed_block_ratio=0.5,
            minimum_segment_samples=3,
        )
        self.assertEqual(len(segments), 1)

    def test_local_transient_change_does_not_create_segment(self) -> None:
        features = [
            *[_feature(index * 2, 0) for index in range(4)],
            _feature(8, 2),
            _feature(10, 0),
            *[_feature((index + 6) * 2, 0) for index in range(4)],
        ]
        segments = find_temporal_segments(
            features,
            confirmation_samples=3,
            changed_block_ratio=0.5,
            minimum_segment_samples=3,
        )
        self.assertEqual(len(segments), 1)

    def test_representative_prefers_later_information_gain(self) -> None:
        features = [
            _feature(0, 0, 0.1),
            _feature(2, 1, 0.2),
            _feature(4, 2, 0.3),
            _feature(6, 3, 0.4),
        ]
        selected = choose_representative_index(features, 0, len(features))
        self.assertGreaterEqual(selected, 2)

    def test_finds_first_persistent_state_crossover(self) -> None:
        before = _feature(0, 0)
        after = _feature(10, 8)
        samples = [
            _feature(0, 0),
            _feature(1, 0),
            _feature(2, 8),
            _feature(3, 8),
            _feature(4, 8),
        ]
        crossover = find_state_crossover(
            samples,
            before_state=before,
            after_state=after,
            persistence=2,
        )
        self.assertEqual(crossover, 2)

    def test_incremental_segmentation_matches_batch_result(self) -> None:
        features = [
            *[_feature(index * 2, 0) for index in range(6)],
            *[_feature((index + 6) * 2, 8) for index in range(6)],
            *[_feature((index + 12) * 2, 0) for index in range(6)],
            *[_feature((index + 18) * 2, 10) for index in range(6)],
        ]
        batch = find_temporal_segments(
            features,
            confirmation_samples=3,
            changed_block_ratio=0.5,
            minimum_segment_samples=3,
        )
        segmenter = self._segmenter()
        emitted: list = []
        for start in range(0, len(features), 6):
            final = start + 6 >= len(features)
            new_segments, _ = segmenter.extend(
                features[start : start + 6],
                final=final,
            )
            emitted.extend(new_segments)
        self.assertEqual(emitted, batch)
        self.assertEqual(segmenter.confirmed_segments, batch)

    def test_incremental_segmentation_emits_before_final_chunk(self) -> None:
        features = [
            *[_feature(index * 2, 0) for index in range(6)],
            *[_feature((index + 6) * 2, 8) for index in range(6)],
            *[_feature((index + 12) * 2, 0) for index in range(6)],
            *[_feature((index + 18) * 2, 10) for index in range(6)],
        ]
        segmenter = self._segmenter()
        early, snapshot = segmenter.extend(features[:18], final=False)
        self.assertEqual(len(snapshot), 3)
        self.assertEqual(len(early), 1)
        remaining, _ = segmenter.extend(features[18:], final=True)
        self.assertEqual(len([*early, *remaining]), 4)
