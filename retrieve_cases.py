"""Convenience entry point for the second retrieval-evaluation stage."""

from __future__ import annotations

import sys

from evaluate import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:], forced_stage="retrieve"))
