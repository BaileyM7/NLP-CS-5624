"""Start the tool server.

Usage:
    python -m toolharness.mcp_server --scenarios data/scenarios/hotpotqa.jsonl [more files]
        [--edits data/edits/hotpotqa.jsonl] [--port 8765]
"""
from __future__ import annotations

import argparse
from typing import Sequence

from ..data.fact_edits import missing_edits, read_edits
from ..data.scenario import read_jsonl
from ..injector.faults import CORRUPTORS, build_corruptors
from .registry import EpisodeRegistry
from .server import build_server


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the toolharness MCP server (streamable HTTP).")
    parser.add_argument("--scenarios", nargs="+", required=True, help="scenario JSONL files to serve")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--gt-log",
        default="runs/ground_truth.jsonl",
        help="ground-truth log path (appended to; one row per agent tool call)",
    )
    parser.add_argument(
        "--edits",
        nargs="+",
        default=[],
        help="fact edit JSONL files; required to serve silently_wrong scenarios",
    )
    args = parser.parse_args(argv)

    scenarios = []
    for path in args.scenarios:
        scenarios.extend(read_jsonl(path))

    # Without --edits, silently_wrong episodes are rejected; with it, every answer page must have an edit
    corruptors = CORRUPTORS
    if args.edits:
        fact_edits = read_edits(args.edits)
        missing = missing_edits(scenarios, fact_edits)
        if missing:
            raise SystemExit(f"{len(missing)} silently_wrong answer pages have no fact edit, e.g. {missing[0]}")
        corruptors = build_corruptors(fact_edits)

    server = build_server(EpisodeRegistry(scenarios, gt_log_path=args.gt_log, corruptors=corruptors))
    print(f"Serving {len(scenarios)} scenarios at http://{args.host}:{args.port}/mcp", flush=True)
    print(f"Ground-truth log: {args.gt_log}", flush=True)
    server.run("streamable-http", host=args.host, port=args.port, stateless_http=True, json_response=True)


if __name__ == "__main__":
    main()
