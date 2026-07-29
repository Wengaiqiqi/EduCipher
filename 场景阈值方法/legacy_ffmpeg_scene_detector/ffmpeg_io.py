"""FFmpeg integration for the archived scene detector."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class FFmpegError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoMetadata:
    duration_sec: float
    width: int
    height: int


class FFmpegTools:
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
        completed = subprocess.run(command, capture_output=True, check=False)
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

    def detect_scene_changes(
        self,
        video_path: Path,
        *,
        threshold: float,
        crop_ratios: tuple[float, float, float, float] | None = None,
    ) -> list[float]:
        filters: list[str] = []
        if crop_ratios is not None:
            left, top, right, bottom = crop_ratios
            width_ratio = 1.0 - left - right
            height_ratio = 1.0 - top - bottom
            filters.append(
                (
                    f"crop=iw*{width_ratio:.6f}:ih*{height_ratio:.6f}:"
                    f"iw*{left:.6f}:ih*{top:.6f}"
                )
            )
        filters.extend([f"select=gt(scene\\,{threshold})", "showinfo"])
        filter_graph = ",".join(filters)
        command = [
            self.ffmpeg,
            "-hide_banner",
            "-nostats",
            "-loglevel",
            "info",
            "-i",
            str(video_path),
            "-vf",
            filter_graph,
            "-an",
            "-sn",
            "-f",
            "null",
            "-",
        ]
        completed = subprocess.run(command, capture_output=True, check=False)
        stderr = completed.stderr.decode("utf-8", errors="replace")
        if completed.returncode != 0:
            tail = "\n".join(stderr.splitlines()[-12:])
            raise FFmpegError(f"FFmpeg scene detection failed:\n{tail}")
        timestamps = {
            float(match)
            for match in re.findall(r"pts_time:\s*([0-9]+(?:\.[0-9]+)?)", stderr)
        }
        return sorted(timestamp for timestamp in timestamps if timestamp > 0)

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
        completed = subprocess.run(command, capture_output=True, check=False)
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
