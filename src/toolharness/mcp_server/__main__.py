"""Start the tool server.

Usage:
    python -m toolharness.mcp_server --scenarios data/scenarios/hotpotqa.jsonl [more files] [--port 8765]
"""
from __future__ import annotations

import argparse
from typing import Sequence

from ..data.scenario import read_jsonl
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
    args = parser.parse_args(argv)

    scenarios = []
    for path in args.scenarios:
        scenarios.extend(read_jsonl(path))

    server = build_server(EpisodeRegistry(scenarios, gt_log_path=args.gt_log))
    print(f"Serving {len(scenarios)} scenarios at http://{args.host}:{args.port}/mcp", flush=True)
    print(f"Ground-truth log: {args.gt_log}", flush=True)
    server.run("streamable-http", host=args.host, port=args.port, stateless_http=True, json_response=True)


if __name__ == "__main__":
    main()
