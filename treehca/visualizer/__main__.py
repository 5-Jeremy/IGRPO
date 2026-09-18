"""Run from the repository root with python -m treehca.visualizer."""

import argparse
from pathlib import Path

from .data import discover_files


def main():
    parser = argparse.ArgumentParser(description="Browse tree-structured WebShop JSONL rollouts.")
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    try:
        discover_files(args.log_dir)
        from .app import create_app

        app = create_app(args.log_dir)
    except (OSError, ValueError, ImportError) as exc:
        parser.error(str(exc))
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
