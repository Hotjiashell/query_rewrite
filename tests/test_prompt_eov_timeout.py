import unittest
from unittest.mock import patch

import requests

from promptEov.engine import _llm


class PromptEovTimeoutTests(unittest.TestCase):
    @patch("promptEov.engine.requests.post")
    def test_llm_timeout_is_passed_to_http_request(self, post):
        post.return_value.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        self.assertEqual(
            _llm(
                "prompt",
                model="model",
                base_url="http://llm",
                api_key="key",
                timeout=180,
            ),
            "ok",
        )
        self.assertEqual(post.call_args.kwargs["timeout"], 180.0)

    @patch("promptEov.engine.requests.post")
    def test_timeout_error_contains_configured_seconds(self, post):
        post.side_effect = requests.Timeout("read timeout")
        with self.assertRaisesRegex(requests.Timeout, "180 seconds"):
            _llm(
                "prompt",
                model="model",
                base_url="http://llm",
                api_key="key",
                timeout=180,
            )


if __name__ == "__main__":
    unittest.main()
