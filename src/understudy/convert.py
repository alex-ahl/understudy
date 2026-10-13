"""The trained adapter as a GGUF file for llama-server.

    understudy convert --llama-cpp ../llama.cpp

Runs llama.cpp's convert_lora_to_gguf.py with this Python, so its requirements
(requirements/requirements-convert_lora_to_gguf.txt) go in this environment.
Downloads the base model's config on the first run.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from understudy.train import MODEL


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="understudy convert", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", type=Path, default=Path("outputs/adapter"))
    ap.add_argument("--out", type=Path, default=Path("outputs/adapter.gguf"))
    ap.add_argument("--model", default=MODEL, help="the base the adapter was trained on")
    ap.add_argument("--llama-cpp", type=Path, default=Path("../llama.cpp"), metavar="DIR",
                    help="a llama.cpp checkout (default ../llama.cpp)")
    return ap


def script(args: argparse.Namespace) -> Path:
    return args.llama_cpp / "convert_lora_to_gguf.py"


def command(args: argparse.Namespace) -> list[str]:
    return [sys.executable, str(script(args)), str(args.adapter),
            "--base-model-id", args.model, "--outfile", str(args.out)]


def run(args: argparse.Namespace) -> int:
    if not script(args).exists():
        print(f"no {script(args)}; point --llama-cpp at a llama.cpp checkout", file=sys.stderr)
        return 1
    if not args.adapter.is_dir():
        print(f"no adapter at {args.adapter}; run understudy train first", file=sys.stderr)
        return 1
    return subprocess.run(command(args)).returncode


def main(argv: list[str] | None = None) -> int:
    return run(parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
