"""Archived FFmpeg scene-threshold detector.

This package is intentionally separate from the current temporal detector.
"""

from .config import DetectorConfig
from .pipeline import VideoPageDetector

__all__ = ["DetectorConfig", "VideoPageDetector"]
