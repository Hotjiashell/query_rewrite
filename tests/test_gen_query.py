import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gen_query import (
    BaselineQueryGenerator,
    MethodV1QueryGenerator,
    PromptMultiQueryGenerator,
    PromptQueryGenerator,
    QueryGenerationError,
    LLMConfig,
    create_query_generator,
    load_prompt_template,
    normalize_query_method,
    extract_query,
    extract_queries,
)


class _FakeCompletions:
    def __init__(self, content='```json\n{"query": "会议室预订流程"}\n```'):
        self.calls = []
        self._content = content

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = type("Message", (), {"content": self._content})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice]})()


class _FakeClient:
    def __init__(self, content='```json\n{"query": "会议室预订流程"}\n```'):
        self.completions = _FakeCompletions(content)
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

    def test_extract_queries_from_list(self):
        content = '```json\n{"query": ["电脑坏了怎么修理", "联系资产管理员", "电脑坏了怎么修理"]}\n```'
        self.assertEqual(extract_queries(content), ["电脑坏了怎么修理", "联系资产管理员"])

    def test_extract_queries_accepts_single_string(self):
        content = '```json\n{"query": "  网络故障  "}\n```'
        self.assertEqual(extract_queries(content), ["网络故障"])

    def test_extract_queries_caps_at_max_queries(self):
        content = '```json\n{"query": ["a", "b", "c", "d"]}\n```'
        self.assertEqual(extract_queries(content, max_queries=3), ["a", "b", "c"])

    def test_extract_queries_rejects_non_json_answer(self):
        with self.assertRaises(QueryGenerationError):
            extract_queries("请搜索网络故障")

    def test_extract_queries_rejects_all_empty_entries(self):
        content = '```json\n{"query": ["", "   "]}\n```'
        with self.assertRaises(QueryGenerationError):
            extract_queries(content)

    def test_multi_query_generator_returns_list_and_disables_thinking(self):
        client = _FakeClient('```json\n{"query": ["电脑坏了怎么修理", "联系资产管理员"]}\n```')
        generator = PromptMultiQueryGenerator(client, "test-model", "multi_query")

        self.assertEqual(
            generator.generate_queries("客服：您好\n用户：电脑坏了"),
            ["电脑坏了怎么修理", "联系资产管理员"],
        )
        self.assertEqual(generator.method, "multi_query")
        call = client.completions.calls[0]
        self.assertEqual(
            call["extra_body"],
            {"chat_template_kwargs": {"enable_thinking": False}},
        )

    def test_multi_query_generator_generate_returns_first_query(self):
        client = _FakeClient('```json\n{"query": ["电脑坏了怎么修理", "联系资产管理员"]}\n```')
        generator = PromptMultiQueryGenerator(client, "test-model", "multi_query")

        self.assertEqual(generator.generate("客服：您好\n用户：电脑坏了"), "电脑坏了怎么修理")

    def test_factory_selects_multi_query_generator(self):
        client = _FakeClient('```json\n{"query": ["a", "b"]}\n```')
        config = LLMConfig("http://model", "test-model", "test-key")
        with patch("gen_query.build_openai_client", return_value=client):
            generator = create_query_generator(config, "multi_query")

        self.assertIsInstance(generator, PromptMultiQueryGenerator)
        self.assertEqual(generator.method, "multi_query")

    def test_load_prompt_template_reads_file_with_placeholder(self):
        with tempfile.TemporaryDirectory() as directory:
            prompt_path = Path(directory) / "prompt.txt"
            prompt_path.write_text("请分析：{dialogue}", encoding="utf-8")

            self.assertEqual(load_prompt_template(str(prompt_path)), "请分析：{dialogue}")

    def test_load_prompt_template_rejects_missing_placeholder(self):
        with tempfile.TemporaryDirectory() as directory:
            prompt_path = Path(directory) / "prompt.txt"
            prompt_path.write_text("no placeholder here", encoding="utf-8")

            with self.assertRaises(ValueError):
                load_prompt_template(str(prompt_path))

    def test_load_prompt_template_rejects_missing_file(self):
        with self.assertRaises(ValueError):
            load_prompt_template("/nonexistent/prompt.txt")

    def test_custom_method_uses_loaded_prompt_template(self):
        with tempfile.TemporaryDirectory() as directory:
            prompt_path = Path(directory) / "prompt.txt"
            prompt_path.write_text(
                '自定义提示词，对话如下：\n{dialogue}\n输出```json\n{"query": "..."}\n```',
                encoding="utf-8",
            )
            client = _FakeClient('```json\n{"query": "自定义结果"}\n```')
            config = LLMConfig("http://model", "test-model", "test-key")
            with patch("gen_query.build_openai_client", return_value=client):
                generator = create_query_generator(config, "custom", prompt_file=str(prompt_path))

        self.assertIsInstance(generator, PromptQueryGenerator)
        self.assertEqual(generator.method, "custom")
        self.assertEqual(generator.generate("用户：你好"), "自定义结果")
        call = client.completions.calls[0]
        self.assertIn("用户：你好", call["messages"][0]["content"])
        self.assertIn("自定义提示词", call["messages"][0]["content"])
        self.assertEqual(
            call["extra_body"],
            {"chat_template_kwargs": {"enable_thinking": False}},
        )

    def test_custom_method_requires_prompt_file(self):
        config = LLMConfig("http://model", "test-model", "test-key")
        with patch("gen_query.build_openai_client", return_value=_FakeClient()):
            with self.assertRaises(ValueError):
                create_query_generator(config, "custom")


if __name__ == "__main__":
    unittest.main()
