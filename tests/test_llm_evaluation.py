import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from video_page_detector.llm_evaluation import (
    LLMEvaluationConfig,
    _page_input_fingerprint,
    build_chat_payload,
    evaluate_transcript,
    normalize_page_evaluation,
    parse_chat_completion_json,
    render_evaluation_markdown,
    summarize_evaluations,
)


class LLMEvaluationTests(unittest.TestCase):
    def test_cache_fingerprint_changes_with_provider_and_request_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            screenshot = Path(directory) / "page.jpg"
            Image.new("RGB", (64, 36), "white").save(screenshot)
            page = {"page_id": 1, "utterances": [{"text": "讲解内容"}]}
            base = _page_input_fingerprint(
                page,
                screenshot,
                LLMEvaluationConfig(base_url="https://provider-a.test/v1"),
            )
            variants = [
                LLMEvaluationConfig(base_url="https://provider-b.test/v1"),
                LLMEvaluationConfig(temperature=0.3),
                LLMEvaluationConfig(image_detail="low"),
                LLMEvaluationConfig(response_format_mode="prompt_only"),
            ]
            for config in variants:
                with self.subTest(config=config):
                    self.assertNotEqual(
                        base,
                        _page_input_fingerprint(page, screenshot, config),
                    )

    def test_request_sends_plain_speech_without_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            screenshot = Path(directory) / "page.jpg"
            Image.new("RGB", (64, 36), "white").save(screenshot)
            payload = build_chat_payload(
                {
                    "page_id": 3,
                    "start_sec": 120,
                    "end_sec": 180,
                    "utterances": [
                        {
                            "start_sec": 122.125,
                            "end_sec": 126.75,
                            "text": "第一段讲话。",
                        },
                        {
                            "start_sec": 130,
                            "end_sec": 135,
                            "text": "第二段讲话。",
                        },
                    ],
                },
                screenshot,
                LLMEvaluationConfig(model="vision-model"),
                include_response_format=True,
            )
        user_text = payload["messages"][1]["content"][0]["text"]
        self.assertIn("第一段讲话。", user_text)
        self.assertIn("第二段讲话。", user_text)
        self.assertNotIn("00:", user_text)
        self.assertNotIn("PPT时间", user_text)
        self.assertNotIn("页首过渡", user_text)
        self.assertNotIn("页尾过渡", user_text)
        system_text = payload["messages"][0]["content"]
        self.assertIn('"reason"', system_text)
        self.assertNotIn('"matched_evidence"', system_text)

    def test_detailed_mode_requests_evidence_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            screenshot = Path(directory) / "page.jpg"
            Image.new("RGB", (64, 36), "white").save(screenshot)
            payload = build_chat_payload(
                {
                    "page_id": 1,
                    "utterances": [{"text": "讲解内容。"}],
                },
                screenshot,
                LLMEvaluationConfig(include_evidence=True),
                include_response_format=True,
            )
        system_text = payload["messages"][0]["content"]
        self.assertIn('"matched_evidence"', system_text)
        self.assertIn('"ppt_key_points"', system_text)

    def test_parses_json_after_reasoning_prefix(self) -> None:
        parsed = parse_chat_completion_json(
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "<think>先分析</think>\n"
                                '{"speech_relevance": 80}'
                            )
                        }
                    }
                ]
            }
        )
        self.assertEqual(parsed["speech_relevance"], 80)

    def test_computes_weighted_page_score_in_code(self) -> None:
        result = normalize_page_evaluation(
            {
                "speech_relevance": 90,
                "ppt_coverage": 72,
                "evidence_consistency": 95,
                "ppt_key_points": ["刚体机械能守恒"],
                "speech_key_points": ["讲解机械能守恒"],
                "matched_evidence": [
                    {"ppt": "机械能守恒", "speech": "动能转化为势能"}
                ],
                "unrelated_content": [],
                "reason": "内容对应",
                "confidence": 0.95,
            },
            page={"page_id": 1, "start_sec": 0, "end_sec": 10},
            fingerprint="abc",
        )
        self.assertEqual(result["score"], 86)
        self.assertEqual(result["level"], "明显相关")
        self.assertNotIn("matched_evidence", result)

    def test_detailed_normalization_and_report_include_evidence(self) -> None:
        page = normalize_page_evaluation(
            {
                "speech_relevance": 90,
                "ppt_coverage": 80,
                "evidence_consistency": 100,
                "ppt_key_points": ["牛顿第二定律"],
                "speech_key_points": ["讲解F等于ma"],
                "matched_evidence": [
                    {"ppt": "F=ma", "speech": "F等于ma"}
                ],
                "unrelated_content": [],
                "reason": "内容一致",
                "confidence": 0.95,
            },
            page={"page_id": 2, "start_sec": 0, "end_sec": 10},
            fingerprint="detailed",
            include_evidence=True,
        )
        report = render_evaluation_markdown(
            {
                "video_id": "lesson",
                "model": "vision-model",
                "summary": {
                    "total_pages": 1,
                    "scored_pages": 1,
                    "no_speech_pages": 0,
                    "failed_pages": 0,
                    "strict_overall_score": 89,
                    "association_average_score": 89,
                    "speech_page_coverage_percent": 100,
                },
                "config": {"include_evidence": True},
                "pages": [page],
            }
        )
        self.assertIn("对应证据：", report)
        self.assertIn("PPT：F=ma；讲话：F等于ma", report)

    def test_concise_report_only_keeps_page_score_and_reason(self) -> None:
        report = render_evaluation_markdown(
            {
                "video_id": "lesson",
                "model": "vision-model",
                "summary": {
                    "total_pages": 1,
                    "scored_pages": 1,
                    "no_speech_pages": 0,
                    "failed_pages": 0,
                    "strict_overall_score": 89,
                    "association_average_score": 89,
                    "speech_page_coverage_percent": 100,
                },
                "config": {"include_evidence": False},
                "pages": [
                    {
                        "page_id": 2,
                        "score": 89,
                        "reason": "讲话内容紧密围绕PPT展开。",
                        "matched_evidence": [
                            {"ppt": "旧证据", "speech": "旧讲话"}
                        ],
                    }
                ],
            }
        )
        self.assertIn("## 第 2 页：89分", report)
        self.assertIn("讲话内容紧密围绕PPT展开。", report)
        self.assertNotIn("| 页码 |", report)
        self.assertNotIn("对应证据：", report)

    def test_summary_has_strict_and_spoken_page_averages(self) -> None:
        summary = summarize_evaluations(
            [
                {"status": "scored", "score": 90},
                {"status": "scored", "score": 70},
                {"status": "no_speech", "score": 0},
            ]
        )
        self.assertEqual(summary["strict_overall_score"], 53.33)
        self.assertEqual(summary["association_average_score"], 80.0)
        self.assertEqual(summary["speech_page_coverage_percent"], 66.67)

    def test_concurrent_evaluation_writes_resume_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            screenshot = root / "page.jpg"
            Image.new("RGB", (64, 36), "white").save(screenshot)
            transcript = root / "transcript.json"
            transcript.write_text(
                json.dumps(
                    {
                        "video_id": "lesson",
                        "pages": [
                            {
                                "page_id": 1,
                                "start_sec": 0,
                                "end_sec": 10,
                                "screenshot_path": screenshot.as_posix(),
                                "utterances": [
                                    {
                                        "start_sec": 1,
                                        "end_sec": 4,
                                        "text": "讲解第一页。",
                                    }
                                ],
                            },
                            {
                                "page_id": 2,
                                "start_sec": 10,
                                "end_sec": 20,
                                "screenshot_path": screenshot.as_posix(),
                                "utterances": [],
                            },
                            {
                                "page_id": 3,
                                "start_sec": 20,
                                "end_sec": 30,
                                "screenshot_path": screenshot.as_posix(),
                                "utterances": [
                                    {
                                        "start_sec": 21,
                                        "end_sec": 24,
                                        "text": "讲解第三页。",
                                    }
                                ],
                            },
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            active = 0
            maximum_active = 0
            request_count = 0

            async def requester(_: dict, __: bool) -> dict:
                nonlocal active, maximum_active, request_count
                request_count += 1
                active += 1
                maximum_active = max(maximum_active, active)
                await asyncio.sleep(0.02)
                active -= 1
                return {
                    "speech_relevance": 90,
                    "ppt_coverage": 80,
                    "evidence_consistency": 90,
                    "ppt_key_points": ["知识点"],
                    "speech_key_points": ["讲解"],
                    "matched_evidence": [],
                    "unrelated_content": [],
                    "reason": "相关",
                    "confidence": 0.9,
                }

            config = LLMEvaluationConfig(
                base_url="http://example.test/v1",
                model="vision-model",
                max_concurrency=2,
            )
            output = root / "evaluation"
            first = evaluate_transcript(
                transcript,
                config=config,
                output_dir=output,
                requester=requester,
            )
            self.assertEqual(request_count, 2)
            self.assertEqual(maximum_active, 2)
            self.assertEqual(first["pages"][1]["status"], "no_speech")
            self.assertTrue((output / "pages" / "page_001.json").is_file())
            self.assertTrue((output / "llm_evaluation.json").is_file())
            self.assertTrue((output / "PPT讲话关联度报告.md").is_file())

            second = evaluate_transcript(
                transcript,
                config=config,
                output_dir=output,
                requester=requester,
            )
            self.assertEqual(request_count, 2)
            self.assertTrue(second["summary"]["complete"])


if __name__ == "__main__":
    unittest.main()
