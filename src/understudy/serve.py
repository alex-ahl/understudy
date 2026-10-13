"""llama-server with the base model and the converted adapter, until stopped.

    understudy serve --lora-scale 1.0

Downloads the base model from --base-hf on the first run; --base-gguf serves a
local file instead. A LoRA scale above 1.0 exaggerates the personas.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

BASE_HF = "AI-Sweden-Models/Llama-3-8B-instruct-Q4_K_M-gguf"


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="understudy serve", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", type=Path, default=Path("private/outputs/adapter.gguf"))
    ap.add_argument("--base-hf", default=BASE_HF, metavar="REPO", help="GGUF base on Hugging Face")
    ap.add_argument("--base-gguf", type=Path, metavar="FILE", help="local GGUF base; wins over --base-hf")
    ap.add_argument("--lora-scale", type=float, default=1.0)
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--llama-server", default="llama-server", metavar="PATH")
    return ap


def command(args: argparse.Namespace) -> list[str]:
    base = ["-m", str(args.base_gguf)] if args.base_gguf else ["-hf", args.base_hf]
    return [args.llama_server, *base, "--lora-scaled", f"{args.adapter}:{args.lora_scale}",
            "-c", str(args.context), "--port", str(args.port)]


def run(args: argparse.Namespace) -> int:
    if not shutil.which(args.llama_server):
        print(f"{args.llama_server} not found; install llama.cpp or pass --llama-server", file=sys.stderr)
        return 1
    if not args.adapter.exists():
        print(f"no adapter at {args.adapter}; run understudy convert first", file=sys.stderr)
        return 1
    cmd = command(args)
    os.execvp(cmd[0], cmd)  # Ctrl-C and service managers reach llama-server directly


def main(argv: list[str] | None = None) -> int:
    return run(parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
