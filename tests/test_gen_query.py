import unittest
from unittest.mock import patch

from gen_query import (
    BaselineQueryGenerator,
    MethodV1QueryGenerator,
    QueryGenerationError,
    LLMConfig,
    create_query_generator,
    normalize_query_method,
    extract_query,
)


class _FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = type("Message", (), {"content": '```json\n{"query": "会议室预订流程"}\n```'})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice]})()


class _FakeClient:
    def __init__(self):
        self.completions = _FakeCompletions()
        self.chat = type("Chat", (), {"completions": self.completions})()


class QueryParsingTests(unittest.TestCase):
    def test_extracts_fenced_json(self):
        self.assertEqual(extract_query('prefix\n```json\n{"query": "  网络故障  "}\n```'), "网络故障")

    def test_rejects_non_json_answer(self):
        with self.assertRaises(QueryGenerationError):
            extract_query("请搜索网络故障")

    def test_baseline_disables_thinking(self):
        client = _FakeClient()
        generator = BaselineQueryGenerator(client, "test-model")

        self.assertEqual(generator.generate("客服：您好\n用户：怎么预定会议"), "会议室预订流程")
        call = client.completions.calls[0]
        self.assertEqual(call["model"], "test-model")
        self.assertEqual(
            call["extra_body"],
            {"chat_template_kwargs": {"enable_thinking": False}},
        )
        self.assertIn("怎么预定会议", call["messages"][0]["content"])

    def test_method_v1_uses_improved_prompt_and_disables_thinking(self):
        client = _FakeClient()
        generator = MethodV1QueryGenerator(client, "test-model")

        self.assertEqual(generator.generate("客服：请问您使用哪个软件？\n用户：企业微信"), "会议室预订流程")
        call = client.completions.calls[0]
        self.assertEqual(generator.method, "method_v1")
        self.assertEqual(
            call["extra_body"],
            {"chat_template_kwargs": {"enable_thinking": False}},
        )
        prompt = call["messages"][0]["content"]
        self.assertIn("你是企业内部智能客服", prompt)
        self.assertIn("充分理解用户问题", prompt)
        self.assertIn("企业微信", prompt)

    def test_query_method_aliases_are_normalized(self):
        self.assertEqual(normalize_query_method("METHOD_V1"), "method_v1")
        self.assertEqual(normalize_query_method("METHOD_V1_PROMPT"), "method_v1")
        self.assertEqual(normalize_query_method("method-v1"), "method_v1")
        with self.assertRaises(ValueError):
            normalize_query_method("unknown")

    def test_factory_selects_configured_method_without_network_call(self):
        client = _FakeClient()
        config = LLMConfig("http://model", "test-model", "test-key")
        with patch("gen_query.build_openai_client", return_value=client) as build_client:
            generator = create_query_generator(config, "method_v1")

        self.assertEqual(generator.method, "method_v1")
        build_client.assert_called_once_with(config)


if __name__ == "__main__":
    unittest.main()
