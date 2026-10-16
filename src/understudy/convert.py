"""The trained adapter as a GGUF file for llama-server.

    understudy convert

Runs llama.cpp's convert_lora_to_gguf.py, which the Docker images carry in
/app with its requirements in the system Python. Downloads the base model's
config on the first run.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from understudy.train import MODEL

# where llama.cpp's images put the converter; its requirements live in the system
# Python, apart from understudy's own environment, whose transformers is newer
CONVERTER = Path("/app/convert_lora_to_gguf.py")
PYTHON = "/usr/bin/python3"


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="understudy convert", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--adapter", type=Path, default=Path("private/outputs/adapter"))
    ap.add_argument("--out", type=Path, default=Path("private/outputs/adapter.gguf"))
    ap.add_argument("--model", default=MODEL, help="the base the adapter was trained on")
    return ap


def command(args: argparse.Namespace) -> list[str]:
    return [PYTHON, str(CONVERTER), str(args.adapter), "--base-model-id", args.model, "--outfile", str(args.out)]


def run(args: argparse.Namespace) -> int:
    if not CONVERTER.exists():
        print(f"no {CONVERTER}; run this in the Docker image (see the README)", file=sys.stderr)
        return 1
    if not args.adapter.is_dir():
        print(f"no adapter at {args.adapter}; run understudy train first", file=sys.stderr)
        return 1
    return subprocess.run(command(args), check=False).returncode


def main(argv: list[str] | None = None) -> int:
    return run(parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
