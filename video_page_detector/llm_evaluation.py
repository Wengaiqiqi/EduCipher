from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence


PROMPT_VERSION = "ppt_speech_relevance_v3_optional_evidence"


@dataclass(frozen=True)
class LLMEvaluationConfig:
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    api_key_env: str = "LLM_API_KEY"
    max_concurrency: int = 5
    timeout_sec: float = 120.0
    max_retries: int = 3
    temperature: float = 0.0
    response_format_mode: str = "json_object"
    image_detail: str = "high"
    include_evidence: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "LLMEvaluationConfig":
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(
                f"Unknown LLM evaluation configuration keys: {', '.join(unknown)}"
            )
        config = cls(**dict(data))
        config.validate()
        return config

    @classmethod
    def from_file(cls, path: str | Path) -> "LLMEvaluationConfig":
        config_path = Path(path)
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid LLM evaluation configuration: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("LLM evaluation configuration root must be an object")
        return cls.from_mapping(data)

    def validate(self) -> None:
        if not self.base_url.strip():
            raise ValueError("LLM base_url must not be empty")
        if not self.model.strip():
            raise ValueError("LLM model must not be empty")
        if not self.api_key_env.strip():
            raise ValueError("api_key_env must not be empty")
        if not 1 <= self.max_concurrency <= 10:
            raise ValueError("max_concurrency must be between 1 and 10")
        if self.timeout_sec <= 0:
            raise ValueError("timeout_sec must be positive")
        if self.max_retries < 1:
            raise ValueError("max_retries must be positive")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        if self.response_format_mode not in {"json_object", "prompt_only"}:
            raise ValueError(
                "response_format_mode must be json_object or prompt_only"
            )
        if self.image_detail not in {"auto", "low", "high"}:
            raise ValueError("image_detail must be auto, low, or high")
        if not isinstance(self.include_evidence, bool):
            raise ValueError("include_evidence must be true or false")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ModelRequester = Callable[
    [dict[str, Any], bool],
    Awaitable[dict[str, Any]],
]
ProgressCallback = Callable[[str, int, int], None]
ActivityCallback = Callable[[int, int], None]


def _chat_completions_endpoint(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return normalized + "/chat/completions"


def _resolve_screenshot_path(
    raw_path: str,
    transcript_path: Path,
) -> Path:
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = transcript_path.resolve().parent / candidate
    if not candidate.is_file():
        raise FileNotFoundError(f"PPT screenshot does not exist: {candidate}")
    return candidate


def _image_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    media_type = "image/png" if suffix == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def _page_utterance_text(page: Mapping[str, Any]) -> str:
    utterances = page.get("utterances", [])
    lines: list[str] = []
    for utterance in utterances:
        text = str(utterance.get("text", "")).strip()
        if text:
            lines.append(text)
    return "\n".join(line for line in lines if line.strip())


def _page_input_fingerprint(
    page: Mapping[str, Any],
    screenshot: Path,
    config: LLMEvaluationConfig,
) -> str:
    digest = hashlib.sha256()
    digest.update(PROMPT_VERSION.encode("utf-8"))
    cache_config = {
        "base_url": config.base_url.rstrip("/"),
        "model": config.model,
        "temperature": config.temperature,
        "response_format_mode": config.response_format_mode,
        "image_detail": config.image_detail,
        "include_evidence": config.include_evidence,
    }
    digest.update(
        json.dumps(cache_config, ensure_ascii=False, sort_keys=True).encode("utf-8")
    )
    digest.update(str(page.get("start_sec")).encode("utf-8"))
    digest.update(str(page.get("end_sec")).encode("utf-8"))
    digest.update(_page_utterance_text(page).encode("utf-8"))
    digest.update(screenshot.read_bytes())
    return digest.hexdigest()


def _system_prompt(include_evidence: bool) -> str:
    shared = """
你是课堂教学内容评估员。请根据一张PPT截图和该页展示期间的讲话文字，
判断讲话内容与当前PPT的关联程度。

不要因为老师没有逐字讲完PPT上的全部内容就判为不相关。
明显的语音识别同音字、简繁体或专业术语错字，应结合PPT语境理解，不直接扣分。
讲话文字已经由程序按照当前PPT的起止时间完成归页，不需要推断或输出时间信息。

三个分项均为0到100：
1. speech_relevance：讲话中有多少内容与当前PPT有关。
2. ppt_coverage：PPT的主要知识点有多少得到讲解。
3. evidence_consistency：专业名词、公式、变量、图表、例题条件和结论是否明确对应。

不要输出最终加权分，程序会按：
speech_relevance×60% + ppt_coverage×25% + evidence_consistency×15%
统一计算。
""".strip()
    if not include_evidence:
        return (
            shared
            + "\n\n"
            + """

只返回一个JSON对象，不要输出Markdown代码块。字段必须是：
{
  "speech_relevance": 0,
  "ppt_coverage": 0,
  "evidence_consistency": 0,
  "reason": "用一段简洁完整的中文说明评分原因"
}
""".strip()
        )
    return (
        shared
        + "\n\n"
        + """

先分别提取PPT与讲话的关键点，再列出双方的明确对应证据，最后评分。
只返回一个JSON对象，不要输出Markdown代码块。字段必须是：
{
  "ppt_key_points": ["..."],
  "speech_key_points": ["..."],
  "matched_evidence": [{"ppt": "...", "speech": "..."}],
  "unrelated_content": ["..."],
  "speech_relevance": 0,
  "ppt_coverage": 0,
  "evidence_consistency": 0,
  "reason": "...",
  "confidence": 0.0
}
""".strip()
    )


def build_chat_payload(
    page: Mapping[str, Any],
    screenshot: Path,
    config: LLMEvaluationConfig,
    *,
    include_response_format: bool,
) -> dict[str, Any]:
    page_id = int(page["page_id"])
    speech = _page_utterance_text(page)
    user_text = (
        f"页面编号：{page_id}\n"
        f"讲话内容（已按该PPT的时间区间截取）：\n{speech}"
    )
    payload: dict[str, Any] = {
        "model": config.model,
        "temperature": config.temperature,
        "messages": [
            {
                "role": "system",
                "content": _system_prompt(config.include_evidence),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": _image_data_url(screenshot),
                            "detail": config.image_detail,
                        },
                    },
                ],
            },
        ],
    }
    if include_response_format:
        payload["response_format"] = {"type": "json_object"}
    return payload


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        if first_newline >= 0:
            stripped = stripped[first_newline + 1 :]
        if stripped.endswith("```"):
            stripped = stripped[:-3]
    return stripped.strip()


def parse_chat_completion_json(response: Mapping[str, Any]) -> dict[str, Any]:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("LLM response does not contain message content") from exc
    if isinstance(content, list):
        text = "".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, Mapping)
        )
    else:
        text = str(content)
    try:
        parsed = json.loads(_strip_json_fence(text))
    except json.JSONDecodeError as exc:
        decoder = json.JSONDecoder()
        parsed = None
        for index, character in enumerate(text):
            if character != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                parsed = candidate
                break
        if parsed is None:
            raise ValueError(f"LLM did not return valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("LLM evaluation must be a JSON object")
    return parsed


def _score_value(raw: Mapping[str, Any], key: str) -> int:
    try:
        value = float(raw[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"LLM result has invalid {key}") from exc
    if not 0 <= value <= 100:
        raise ValueError(f"LLM score {key} must be between 0 and 100")
    return int(round(value))


def _string_list(raw: Mapping[str, Any], key: str) -> list[str]:
    value = raw.get(key, [])
    if not isinstance(value, list):
        raise ValueError(f"LLM result field {key} must be an array")
    return [str(item).strip() for item in value if str(item).strip()]


def _evidence_list(raw: Mapping[str, Any]) -> list[dict[str, str]]:
    value = raw.get("matched_evidence", [])
    if not isinstance(value, list):
        raise ValueError("matched_evidence must be an array")
    evidence: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        ppt = str(item.get("ppt", "")).strip()
        speech = str(item.get("speech", "")).strip()
        if ppt or speech:
            evidence.append({"ppt": ppt, "speech": speech})
    return evidence


def score_level(score: int) -> str:
    if score >= 90:
        return "高度相关"
    if score >= 75:
        return "明显相关"
    if score >= 60:
        return "部分相关"
    if score >= 40:
        return "弱相关"
    return "基本无关"


def normalize_page_evaluation(
    raw: Mapping[str, Any],
    *,
    page: Mapping[str, Any],
    fingerprint: str,
    include_evidence: bool = False,
) -> dict[str, Any]:
    speech_relevance = _score_value(raw, "speech_relevance")
    ppt_coverage = _score_value(raw, "ppt_coverage")
    evidence_consistency = _score_value(raw, "evidence_consistency")
    score = int(
        round(
            speech_relevance * 0.60
            + ppt_coverage * 0.25
            + evidence_consistency * 0.15
        )
    )
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    result: dict[str, Any] = {
        "page_id": int(page["page_id"]),
        "start_sec": round(float(page["start_sec"]), 3),
        "end_sec": round(float(page["end_sec"]), 3),
        "status": "scored",
        "input_fingerprint": fingerprint,
        "speech_relevance": speech_relevance,
        "ppt_coverage": ppt_coverage,
        "evidence_consistency": evidence_consistency,
        "score": score,
        "level": score_level(score),
        "reason": str(raw.get("reason", "")).strip(),
    }
    if include_evidence:
        result.update(
            {
                "ppt_key_points": _string_list(raw, "ppt_key_points"),
                "speech_key_points": _string_list(raw, "speech_key_points"),
                "matched_evidence": _evidence_list(raw),
                "unrelated_content": _string_list(raw, "unrelated_content"),
                "confidence": round(min(1.0, max(0.0, confidence)), 4),
            }
        )
    return result


def _no_speech_evaluation(
    page: Mapping[str, Any],
    fingerprint: str,
    include_evidence: bool = False,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "page_id": int(page["page_id"]),
        "start_sec": round(float(page["start_sec"]), 3),
        "end_sec": round(float(page["end_sec"]), 3),
        "status": "no_speech",
        "input_fingerprint": fingerprint,
        "speech_relevance": 0,
        "ppt_coverage": 0,
        "evidence_consistency": 0,
        "score": 0,
        "level": "无讲话",
        "reason": "该页面时间段内没有识别到讲话，无法判断内容关联度。",
    }
    if include_evidence:
        result.update(
            {
                "ppt_key_points": [],
                "speech_key_points": [],
                "matched_evidence": [],
                "unrelated_content": [],
                "confidence": 1.0,
            }
        )
    return result


async def _http_requester(
    config: LLMEvaluationConfig,
    api_key: str,
    payload: dict[str, Any],
    include_response_format: bool,
) -> dict[str, Any]:
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError(
            "缺少HTTP依赖，请执行：python -m pip install httpx"
        ) from exc
    request_payload = dict(payload)
    if not include_response_format:
        request_payload.pop("response_format", None)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=config.timeout_sec) as client:
        response = await client.post(
            _chat_completions_endpoint(config.base_url),
            headers=headers,
            json=request_payload,
        )
        if response.is_error:
            raise RuntimeError(
                f"LLM HTTP {response.status_code}: {response.text[:1000]}"
            )
        return parse_chat_completion_json(response.json())


async def _request_with_retries(
    config: LLMEvaluationConfig,
    api_key: str,
    payload: dict[str, Any],
    requester: ModelRequester | None,
) -> dict[str, Any]:
    include_response_format = config.response_format_mode == "json_object"
    last_error: Exception | None = None
    for attempt in range(1, config.max_retries + 1):
        try:
            if requester is not None:
                return await requester(payload, include_response_format)
            return await _http_requester(
                config,
                api_key,
                payload,
                include_response_format,
            )
        except Exception as exc:
            last_error = exc
            message = str(exc).lower()
            if include_response_format and (
                "response_format" in message
                or "json_object" in message
                or "unsupported" in message
            ):
                include_response_format = False
                continue
            if attempt < config.max_retries:
                delay = min(8.0, 2 ** (attempt - 1)) + random.random() * 0.25
                await asyncio.sleep(delay)
    assert last_error is not None
    raise last_error


def _load_transcript(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid transcript JSON: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("pages"), list):
        raise ValueError("transcript.json must contain a pages array")
    pages = [dict(page) for page in payload["pages"] if isinstance(page, Mapping)]
    if not pages:
        raise ValueError("transcript.json contains no pages")
    return payload, pages


def _load_cached_page(path: Path, fingerprint: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if (
        isinstance(payload, dict)
        and payload.get("input_fingerprint") == fingerprint
        and payload.get("status") in {"scored", "no_speech"}
    ):
        return payload
    return None


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def summarize_evaluations(
    evaluations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    total_pages = len(evaluations)
    scored = [item for item in evaluations if item.get("status") == "scored"]
    no_speech = [
        item for item in evaluations if item.get("status") == "no_speech"
    ]
    failed = [item for item in evaluations if item.get("status") == "failed"]
    score_sum = sum(float(item.get("score", 0)) for item in scored)
    strict_overall = (
        round(score_sum / total_pages, 2)
        if total_pages and not failed
        else None
    )
    association_average = (
        round(score_sum / len(scored), 2)
        if scored
        else None
    )
    coverage = (
        round(len(scored) / total_pages * 100.0, 2)
        if total_pages
        else 0.0
    )
    return {
        "total_pages": total_pages,
        "scored_pages": len(scored),
        "no_speech_pages": len(no_speech),
        "failed_pages": len(failed),
        "complete": not failed,
        "score_sum": round(score_sum, 2),
        "strict_overall_score": strict_overall,
        "association_average_score": association_average,
        "speech_page_coverage_percent": coverage,
        "strict_overall_level": (
            score_level(int(round(strict_overall)))
            if strict_overall is not None
            else None
        ),
    }


def render_evaluation_markdown(result: Mapping[str, Any]) -> str:
    summary = result["summary"]
    include_evidence = bool(
        result.get("config", {}).get("include_evidence", False)
    )
    lines = [
        f"# {result['video_id']} PPT与讲话关联度评估",
        "",
        f"- 模型：`{result['model']}`",
        f"- PPT总页数：{summary['total_pages']}",
        f"- 已评分页面：{summary['scored_pages']}",
        f"- 无讲话页面：{summary['no_speech_pages']}",
        f"- 失败页面：{summary['failed_pages']}",
        (
            f"- 严格总分：{summary['strict_overall_score']}"
            if summary["strict_overall_score"] is not None
            else "- 严格总分：未完成，存在请求失败页面"
        ),
        f"- 纯关联平均分：{summary['association_average_score']}",
        f"- 讲话页面覆盖率：{summary['speech_page_coverage_percent']}%",
    ]
    if include_evidence:
        lines.extend(
            [
                "- 详细对应证据：已开启",
                "",
                "| 页码 | 状态 | 讲话相关度 | PPT覆盖度 | "
                "证据一致性 | 页面分数 | 等级 |",
                "|---:|---|---:|---:|---:|---:|---|",
            ]
        )
        for page in result["pages"]:
            lines.append(
                "| {page_id} | {status} | {speech_relevance} | "
                "{ppt_coverage} | {evidence_consistency} | "
                "{score} | {level} |".format(**page)
            )
    for page in result["pages"]:
        lines.extend(
            [
                "",
                f"## 第 {page['page_id']} 页：{page['score']}分",
                "",
                page.get("reason", "") or "（没有说明）",
                "",
            ]
        )
        evidence = page.get("matched_evidence", []) if include_evidence else []
        if evidence:
            lines.append("对应证据：")
            lines.append("")
            for item in evidence:
                lines.append(
                    f"- PPT：{item.get('ppt', '')}；讲话：{item.get('speech', '')}"
                )
    return "\n".join(lines).rstrip() + "\n"


async def evaluate_page_async(
    page: Mapping[str, Any],
    *,
    transcript_path: str | Path,
    config: LLMEvaluationConfig,
    output_dir: str | Path,
    api_key: str,
    requester: ModelRequester | None = None,
) -> dict[str, Any]:
    config.validate()
    transcript_file = Path(transcript_path)
    destination = Path(output_dir)
    page_dir = destination / "pages"
    page_dir.mkdir(parents=True, exist_ok=True)
    page_id = int(page["page_id"])
    raw_screenshot = page.get("screenshot_path")
    if not raw_screenshot:
        raise ValueError(f"page {page_id} has no screenshot_path")
    screenshot = _resolve_screenshot_path(
        str(raw_screenshot),
        transcript_file,
    )
    fingerprint = _page_input_fingerprint(page, screenshot, config)
    cache_path = page_dir / f"page_{page_id:03d}.json"
    cached = _load_cached_page(cache_path, fingerprint)
    if cached is not None:
        return cached
    if not _page_utterance_text(page):
        result = _no_speech_evaluation(page, fingerprint, include_evidence=config.include_evidence)
        _write_json(cache_path, result)
        return result
    try:
        payload = build_chat_payload(
            page,
            screenshot,
            config,
            include_response_format=(
                config.response_format_mode == "json_object"
            ),
        )
        raw = await _request_with_retries(
            config,
            api_key,
            payload,
            requester,
        )
        result = normalize_page_evaluation(
            raw,
            page=page,
            fingerprint=fingerprint,
            include_evidence=config.include_evidence,
        )
    except Exception as exc:
        result = {
            "page_id": page_id,
            "start_sec": round(float(page["start_sec"]), 3),
            "end_sec": round(float(page["end_sec"]), 3),
            "status": "failed",
            "input_fingerprint": fingerprint,
            "speech_relevance": 0,
            "ppt_coverage": 0,
            "evidence_consistency": 0,
            "score": 0,
            "level": "请求失败",
            "reason": str(exc),
        }
        if config.include_evidence:
            result["matched_evidence"] = []
    _write_json(cache_path, result)
    return result


def evaluate_page(
    page: Mapping[str, Any],
    *,
    transcript_path: str | Path,
    config: LLMEvaluationConfig,
    output_dir: str | Path,
    api_key: str,
    requester: ModelRequester | None = None,
) -> dict[str, Any]:
    return asyncio.run(
        evaluate_page_async(
            page,
            transcript_path=transcript_path,
            config=config,
            output_dir=output_dir,
            api_key=api_key,
            requester=requester,
        )
    )


async def evaluate_transcript_async(
    transcript_path: str | Path,
    *,
    config: LLMEvaluationConfig,
    output_dir: str | Path | None = None,
    api_key: str | None = None,
    progress_callback: ProgressCallback | None = None,
    activity_callback: ActivityCallback | None = None,
    requester: ModelRequester | None = None,
) -> dict[str, Any]:
    config.validate()
    transcript_file = Path(transcript_path)
    if not transcript_file.is_file():
        raise FileNotFoundError(f"Transcript file does not exist: {transcript_file}")
    transcript, pages = _load_transcript(transcript_file)
    destination = (
        Path(output_dir)
        if output_dir is not None
        else transcript_file.resolve().parent / "llm_evaluation"
    )
    (destination / "pages").mkdir(parents=True, exist_ok=True)
    resolved_api_key = api_key or os.environ.get(config.api_key_env, "")
    if requester is None and not resolved_api_key:
        raise ValueError(
            f"未找到LLM密钥。请设置环境变量 {config.api_key_env}，"
            "或在GUI中临时输入密钥。"
        )

    semaphore = asyncio.Semaphore(config.max_concurrency)
    completed = 0
    completed_lock = asyncio.Lock()
    active = 0
    active_lock = asyncio.Lock()

    async def report_activity(delta: int) -> None:
        nonlocal active
        async with active_lock:
            active = max(0, active + delta)
            current = active
        if activity_callback is not None:
            activity_callback(current, config.max_concurrency)

    async def report(message: str) -> None:
        nonlocal completed
        async with completed_lock:
            completed += 1
            if progress_callback is not None:
                progress_callback(message, completed, len(pages))

    async def evaluate_page_for_batch(
        page: dict[str, Any],
    ) -> dict[str, Any]:
        async with semaphore:
            await report_activity(1)
            try:
                result = await evaluate_page_async(
                    page,
                    transcript_path=transcript_file,
                    config=config,
                    output_dir=destination,
                    api_key=resolved_api_key,
                    requester=requester,
                )
            finally:
                await report_activity(-1)
        await report(f"第 {page['page_id']} 页：{result['status']}")
        return result

    started_at = time.perf_counter()
    results = await asyncio.gather(
        *(evaluate_page_for_batch(page) for page in pages)
    )
    ordered = sorted(results, key=lambda item: int(item["page_id"]))
    summary = summarize_evaluations(ordered)
    final_result: dict[str, Any] = {
        "video_id": str(transcript.get("video_id", transcript_file.parent.name)),
        "transcript_path": transcript_file.resolve().as_posix(),
        "model": config.model,
        "base_url": config.base_url,
        "prompt_version": PROMPT_VERSION,
        "processing_duration_sec": round(time.perf_counter() - started_at, 3),
        "summary": summary,
        "pages": ordered,
        "config": {
            key: value
            for key, value in config.to_dict().items()
            if key != "api_key_env"
        },
    }
    result_path = destination / "llm_evaluation.json"
    report_path = destination / "PPT讲话关联度报告.md"
    final_result["artifacts"] = {
        "result_json": result_path.resolve().as_posix(),
        "report_markdown": report_path.resolve().as_posix(),
        "page_results_dir": (destination / "pages").resolve().as_posix(),
    }
    _write_json(result_path, final_result)
    report_path.write_text(
        render_evaluation_markdown(final_result),
        encoding="utf-8",
    )
    return final_result


def evaluate_transcript(
    transcript_path: str | Path,
    *,
    config: LLMEvaluationConfig,
    output_dir: str | Path | None = None,
    api_key: str | None = None,
    progress_callback: ProgressCallback | None = None,
    activity_callback: ActivityCallback | None = None,
    requester: ModelRequester | None = None,
) -> dict[str, Any]:
    return asyncio.run(
        evaluate_transcript_async(
            transcript_path,
            config=config,
            output_dir=output_dir,
            api_key=api_key,
            progress_callback=progress_callback,
            activity_callback=activity_callback,
            requester=requester,
        )
    )
