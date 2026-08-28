import unittest

from gen_query import BaselineQueryGenerator, QueryGenerationError, extract_query


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


if __name__ == "__main__":
    unittest.main()
