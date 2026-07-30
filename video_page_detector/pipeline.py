"""Default production PPT detector.

The public ``VideoPageDetector`` uses incremental temporal confirmation so
confirmed pages can enter ASR and LLM processing before video analysis ends.
The previous whole-video implementation remains available as
``batch_pipeline.BatchVideoPageDetector`` for regression comparisons.
"""

from .streaming_pipeline import VideoPageDetector

__all__ = ["VideoPageDetector"]
