from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Windows: prevent ffmpeg subprocess from inheriting Worker's stdin pipe
# (which Rust holds open), and match the CREATE_NO_WINDOW flag Rust uses.
_SUBPROCESS_KWARGS = (
    {"stdin": subprocess.DEVNULL, "creationflags": subprocess.CREATE_NO_WINDOW}
    if sys.platform == "win32"
    else {"stdin": subprocess.DEVNULL}
)

import numpy as np


class FFmpegError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoMetadata:
    duration_sec: float
    width: int
    height: int


class FFmpegTools:
    """Generic FFmpeg decoding tools used by temporal detection.

    Scene-threshold detection lives in ``legacy_ffmpeg_scene_detector``.
    """

    def __init__(
        self,
        *,
        ffmpeg_path: str | None,
        ffprobe_path: str | None,
        analysis_width: int,
        analysis_height: int,
    ) -> None:
        self.ffmpeg = self._resolve(ffmpeg_path, "ffmpeg")
        self.ffprobe = self._resolve(ffprobe_path, "ffprobe")
        self.analysis_width = analysis_width
        self.analysis_height = analysis_height

    @staticmethod
    def _resolve(configured: str | None, command: str) -> str:
        if configured:
            path = Path(configured)
            if not path.is_file():
                raise FFmpegError(f"Configured {command} does not exist: {path}")
            return str(path)
        discovered = shutil.which(command)
        if not discovered:
            raise FFmpegError(
                f"{command} was not found. Install FFmpeg and ensure {command} "
                "is on PATH, or set its path in the configuration file."
            )
        return discovered

    def probe(self, video_path: Path) -> VideoMetadata:
        command = [
            self.ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height:format=duration",
            "-of",
            "json",
            str(video_path),
        ]
        try:
            completed = subprocess.run(command, capture_output=True, check=False, timeout=30, **_SUBPROCESS_KWARGS)
        except subprocess.TimeoutExpired:
            raise FFmpegError("ffprobe 读取视频超时(30秒)，请检查视频文件是否可访问")
        if completed.returncode != 0:
            message = completed.stderr.decode("utf-8", errors="replace").strip()
            raise FFmpegError(f"ffprobe failed: {message}")
        try:
            payload = json.loads(completed.stdout.decode("utf-8"))
            stream = payload["streams"][0]
            duration = float(payload["format"]["duration"])
            width = int(stream["width"])
            height = int(stream["height"])
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise FFmpegError("ffprobe returned incomplete video metadata") from exc
        if duration <= 0:
            raise FFmpegError("video duration must be positive")
        return VideoMetadata(duration_sec=duration, width=width, height=height)

    def sample_window(
        self,
        video_path: Path,
        *,
        start_sec: float,
        duration_sec: float,
        fps: float,
    ) -> list[np.ndarray]:
        if duration_sec <= 0 or fps <= 0:
            raise ValueError("duration and fps must be positive")
        command = [
            self.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{max(0.0, start_sec):.6f}",
            "-i",
            str(video_path),
            "-t",
            f"{duration_sec:.6f}",
            "-vf",
            (
                f"fps={fps},scale={self.analysis_width}:"
                f"{self.analysis_height}:flags=fast_bilinear"
            ),
            "-an",
            "-sn",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
        try:
            completed = subprocess.run(command, capture_output=True, check=False, timeout=60, **_SUBPROCESS_KWARGS)
        except subprocess.TimeoutExpired:
            raise FFmpegError("FFmpeg 提取帧超时(60秒)，请检查视频文件是否可访问")
        if completed.returncode != 0:
            message = completed.stderr.decode("utf-8", errors="replace").strip()
            raise FFmpegError(f"FFmpeg frame extraction failed: {message}")
        frame_size = self.analysis_width * self.analysis_height * 3
        frame_count, remainder = divmod(len(completed.stdout), frame_size)
        if remainder:
            raise FFmpegError("FFmpeg returned a truncated raw video frame")
        if frame_count == 0:
            return []
        array = np.frombuffer(completed.stdout, dtype=np.uint8)
        reshaped = array.reshape(
            frame_count,
            self.analysis_height,
            self.analysis_width,
            3,
        )
        return [frame.copy() for frame in reshaped]
