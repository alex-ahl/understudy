"""llama-server with the base model and the converted adapter, until stopped.

    understudy serve --lora-scale 1.0

Downloads the base model from --base-hf on the first run; --base-gguf serves a
local file instead. A LoRA scale above 1.0 exaggerates the personas. Every layer
goes to the GPU when the image has one.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

BASE_HF = "AI-Sweden-Models/Llama-3-8B-instruct-Q4_K_M-gguf"
LLAMA_SERVER = "llama-server"  # on PATH in the Docker images


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="understudy serve", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--adapter", type=Path, default=Path("private/outputs/adapter.gguf"))
    ap.add_argument("--base-hf", default=BASE_HF, metavar="REPO", help="GGUF base on Hugging Face")
    ap.add_argument("--base-gguf", type=Path, metavar="FILE", help="local GGUF base; wins over --base-hf")
    ap.add_argument("--lora-scale", type=float, default=1.0)
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--host", default="0.0.0.0", help="address to listen on (default: all, for Docker)")
    ap.add_argument("--port", type=int, default=8080)
    return ap


def command(args: argparse.Namespace) -> list[str]:
    base = ["-m", str(args.base_gguf)] if args.base_gguf else ["-hf", args.base_hf]
    # -ngl 99: all layers on the GPU; a CPU build ignores it
    return [
        LLAMA_SERVER,
        *base,
        "--lora-scaled",
        f"{args.adapter}:{args.lora_scale}",
        "-c",
        str(args.context),
        "-ngl",
        "99",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]


def run(args: argparse.Namespace) -> int:
    if not args.adapter.exists():
        print(f"no adapter at {args.adapter}; run understudy convert first", file=sys.stderr)
        return 1
    cmd = command(args)
    os.execvp(cmd[0], cmd)  # Ctrl-C and service managers reach llama-server directly


def main(argv: list[str] | None = None) -> int:
    return run(parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
