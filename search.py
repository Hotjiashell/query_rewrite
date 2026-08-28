"""Client for the case-retrieval service.

The evaluator imports :func:`test_retrieval` from this module so the retrieval
endpoint remains centralized here.  The function name is retained for
backwards compatibility with existing ad-hoc scripts.
"""

from __future__ import annotations

from typing import Any

import requests


DEFAULT_RETRIEVAL_URL = "http://10.67.43.14:8276/run_case_retrieval"


class RetrievalRequestError(RuntimeError):
    """Raised when the retrieval service cannot return a usable response."""


def test_retrieval(
    query: str,
    *,
    url: str = DEFAULT_RETRIEVAL_URL,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Retrieve cases for ``query`` from the configured service endpoint.

    ``timeout`` is deliberately required by the implementation so a stalled
    service cannot hold an evaluation worker forever.  HTTP and JSON errors
    are converted to :class:`RetrievalRequestError` for the evaluator to
    record per sample.
    """

    if not query or not query.strip():
        raise ValueError("query must not be empty")

    try:
        response = requests.post(
            url,
            json={"retrieval_query": query},
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise RetrievalRequestError(f"retrieval request failed: {exc}") from exc
    except ValueError as exc:
        raise RetrievalRequestError("retrieval response is not valid JSON") from exc

    if not isinstance(payload, dict):
        raise RetrievalRequestError("retrieval response JSON must be an object")
    return payload

# 正常返回的response.json()格式
"""
{
    "retrieval_result": {
        "top1": {
            "case_id": KTXXXXXXXX,
            "case_title": "案例标题1",
            "content": "案例内容1",
            "score": 0.95
        },
        "top2": {
            "case_id": KTXXXXXXXX,
            "case_title": "案例标题2",
            "content": "案例内容2",
            "score": 0.90
        },
        ...
        "topN": {
            "case_id": KTXXXXXXXX,
            "case_title": "案例标题N",
            "content": "案例内容N",
            "score": 0.85
        }
    }
}
"""
