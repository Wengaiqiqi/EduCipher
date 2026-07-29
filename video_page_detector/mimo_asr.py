from __future__ import annotations

import base64
import os
import random
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True)
class MimoASRSettings:
    base_url: str = "https://api.xiaomimimo.com/v1"
    model: str = "mimo-v2.5-asr"
    language: str = "auto"
    max_concurrency: int = 3
    timeout_sec: float = 180.0
    max_retries: int = 3
    ffmpeg_path: str | None = None

    def validate(self) -> None:
        if not self.base_url.strip():
            raise ValueError("小米ASR Base URL不能为空。")
        if not self.model.strip():
            raise ValueError("小米ASR模型名称不能为空。")
        if not self.language.strip():
            raise ValueError("小米ASR语言设置不能为空。")
        if not 1 <= self.max_concurrency <= 10:
            raise ValueError("小米ASR并发数量必须在1到10之间。")
        if self.timeout_sec <= 0:
            raise ValueError("小米ASR超时时间必须大于0。")
        if self.max_retries < 1:
            raise ValueError("小米ASR重试次数必须大于0。")


@dataclass(frozen=True)
class MimoPageResult:
    page_id: int
    start_sec: float
    end_sec: float
    text: str
    request_duration_sec: float
    audio_size_bytes: int


MimoProgressCallback = Callable[[str, int, int], None]
MimoRequester = Callable[[Path], str]
AudioExtractor = Callable[[Path, float, float, Path], None]


def chat_completions_endpoint(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return normalized + "/chat/completions"


def resolve_ffmpeg(configured: str | None) -> str:
    if configured:
        candidate = Path(configured)
        if not candidate.is_file():
            raise FileNotFoundError(f"配置的FFmpeg不存在：{candidate}")
        return str(candidate)
    discovered = shutil.which("ffmpeg")
    if not discovered:
        raise FileNotFoundError(
            "没有找到FFmpeg。请安装FFmpeg并加入PATH，或在配置中指定路径。"
        )
    return discovered


def extract_wav_segment(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    output_path: Path,
    *,
    ffmpeg_path: str | None = None,
) -> None:
    duration = end_sec - start_sec
    if start_sec < 0 or duration <= 0:
        raise ValueError("音频片段时间范围无效。")
    command = [
        resolve_ffmpeg(ffmpeg_path),
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_sec:.6f}",
        "-i",
        str(video_path),
        "-t",
        f"{duration:.6f}",
        "-vn",
        "-sn",
        "-dn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        details = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"FFmpeg提取临时音频失败：{details}")
    if not output_path.is_file() or output_path.stat().st_size <= 44:
        raise RuntimeError("没有从视频中提取到有效音频。")


def extract_chat_text(payload: Mapping[str, Any]) -> str:
    try:
        message = payload["choices"][0]["message"]
        content = message["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("小米ASR响应缺少 choices[0].message.content。") from exc
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, Mapping):
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        if parts:
            return "\n".join(parts)
    raise ValueError("小米ASR响应中的文字格式无法识别。")


def request_mimo_asr(
    audio_path: Path,
    *,
    settings: MimoASRSettings,
    api_key: str,
) -> str:
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError(
            "缺少HTTP依赖，请执行：python -m pip install httpx"
        ) from exc
    audio_base64 = base64.b64encode(audio_path.read_bytes()).decode("ascii")
    payload = {
        "model": settings.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": f"data:audio/wav;base64,{audio_base64}"
                        },
                    }
                ],
            }
        ],
        "asr_options": {"language": settings.language},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = httpx.post(
        chat_completions_endpoint(settings.base_url),
        headers=headers,
        json=payload,
        timeout=settings.timeout_sec,
    )
    if response.is_error:
        raise RuntimeError(
            f"小米ASR HTTP {response.status_code}：{response.text[:1000]}"
        )
    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError("小米ASR返回了无效JSON。") from exc
    if not isinstance(data, Mapping):
        raise RuntimeError("小米ASR响应根节点不是对象。")
    return extract_chat_text(data)


def _request_with_retries(
    audio_path: Path,
    *,
    settings: MimoASRSettings,
    requester: MimoRequester,
) -> str:
    last_error: Exception | None = None
    for attempt in range(1, settings.max_retries + 1):
        try:
            return requester(audio_path)
        except Exception as exc:
            last_error = exc
            if attempt < settings.max_retries:
                time.sleep(
                    min(8.0, 2 ** (attempt - 1)) + random.random() * 0.25
                )
    assert last_error is not None
    raise last_error


def transcribe_pages_with_mimo(
    video_path: str | Path,
    pages: Sequence[Mapping[str, Any]],
    *,
    settings: MimoASRSettings,
    api_key: str,
    progress_callback: MimoProgressCallback | None = None,
    requester: MimoRequester | None = None,
    audio_extractor: AudioExtractor | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    settings.validate()
    if not api_key.strip():
        raise ValueError("小米ASR API Key不能为空。")
    video = Path(video_path)
    if not video.is_file():
        raise FileNotFoundError(f"视频文件不存在：{video}")
    total = len(pages)
    if not total:
        raise ValueError("没有可转写的PPT页面。")

    actual_requester = requester or (
        lambda audio_path: request_mimo_asr(
            audio_path,
            settings=settings,
            api_key=api_key,
        )
    )
    actual_extractor = audio_extractor or (
        lambda source, start, end, target: extract_wav_segment(
            source,
            start,
            end,
            target,
            ffmpeg_path=settings.ffmpeg_path,
        )
    )
    progress_lock = threading.Lock()
    completed_count = 0

    def report(message: str) -> None:
        nonlocal completed_count
        if progress_callback is None:
            return
        with progress_lock:
            completed_count += 1
            progress_callback(message, completed_count, total)

    def process_page(
        page: Mapping[str, Any],
        temp_root: Path,
    ) -> MimoPageResult:
        page_id = int(page["page_id"])
        start_sec = float(page["start_sec"])
        end_sec = float(page["end_sec"])
        audio_path = temp_root / f"page_{page_id:04d}.wav"
        request_started = time.perf_counter()
        try:
            actual_extractor(video, start_sec, end_sec, audio_path)
            audio_size = audio_path.stat().st_size
            text = _request_with_retries(
                audio_path,
                settings=settings,
                requester=actual_requester,
            ).strip()
            return MimoPageResult(
                page_id=page_id,
                start_sec=start_sec,
                end_sec=end_sec,
                text=text,
                request_duration_sec=time.perf_counter() - request_started,
                audio_size_bytes=audio_size,
            )
        finally:
            try:
                audio_path.unlink(missing_ok=True)
            except OSError:
                pass

    results: dict[int, MimoPageResult] = {}
    with tempfile.TemporaryDirectory(prefix="classroom-ppt-mimo-asr-") as temp:
        temp_root = Path(temp)
        with ThreadPoolExecutor(
            max_workers=min(settings.max_concurrency, total),
            thread_name_prefix="mimo-asr",
        ) as executor:
            futures: dict[Future[MimoPageResult], int] = {}
            for page in pages:
                future = executor.submit(process_page, page, temp_root)
                futures[future] = int(page["page_id"])
            for future in as_completed(futures):
                page_id = futures[future]
                result = future.result()
                results[page_id] = result
                report(f"第{page_id}页云端语音识别完成")

    page_transcripts: list[dict[str, Any]] = []
    request_durations: list[float] = []
    uploaded_bytes = 0
    for page in pages:
        page_id = int(page["page_id"])
        item = results[page_id]
        request_durations.append(item.request_duration_sec)
        uploaded_bytes += item.audio_size_bytes
        utterances = (
            [
                {
                    "start_sec": round(item.start_sec, 3),
                    "end_sec": round(item.end_sec, 3),
                    "text": item.text,
                }
            ]
            if item.text
            else []
        )
        page_transcripts.append(
            {
                "page_id": page_id,
                "start_sec": round(item.start_sec, 3),
                "end_sec": round(item.end_sec, 3),
                "screenshot_path": page.get("screenshot_path"),
                "speech_text": item.text,
                "utterances": utterances,
                "cloud_request_duration_sec": round(
                    item.request_duration_sec, 3
                ),
            }
        )
    statistics = {
        "page_request_count": total,
        "max_concurrency": min(settings.max_concurrency, total),
        "uploaded_audio_bytes": uploaded_bytes,
        "average_page_request_duration_sec": round(
            sum(request_durations) / len(request_durations), 3
        ),
        "slowest_page_request_duration_sec": round(
            max(request_durations), 3
        ),
    }
    return page_transcripts, statistics


def resolve_mimo_api_key(
    configured_env: str,
    explicit_api_key: str | None = None,
) -> str:
    if explicit_api_key and explicit_api_key.strip():
        return explicit_api_key.strip()
    names = [configured_env, "MIMO_API_KEY", "LLM_API_KEY"]
    for name in dict.fromkeys(item for item in names if item):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    raise ValueError(
        "没有找到小米ASR API Key。请在界面输入，或设置 MIMO_API_KEY。"
    )
