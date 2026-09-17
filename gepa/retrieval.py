"""Adapter for the existing retrieval API, preserving variable-length topN results."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate import extract_retrieval_trace  # noqa: E402
from search import test_retrieval  # noqa: E402


@dataclass(frozen=True)
class RetrievedCase:
    rank: int
    case_id: str
    case_title: str


class CaseRetriever:
    def __init__(self, url: str, timeout: float, top_k: int) -> None:
        self.url, self.timeout, self.top_k = url, timeout, top_k

    def retrieve(self, query: str) -> list[RetrievedCase]:
        response: Mapping[str, Any] = test_retrieval(query, url=self.url, timeout=self.timeout)
        # extract_retrieval_trace sorts every key matching top<integer>; it does
        # not assume that the API has returned a contiguous top1..topN range.
        return [RetrievedCase(item.rank, item.case_id, item.case_title) for item in extract_retrieval_trace(response)[: self.top_k]]


def matched_rank(expected_case_id: str, cases: list[RetrievedCase]) -> int | None:
    return next((case.rank for case in cases if case.case_id == expected_case_id), None)
