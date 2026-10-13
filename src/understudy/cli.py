"""understudy [--config FILE] <command> [args]: each command takes its module's own flags (-h).

People and settings come from FILE, or private/understudy.toml if it exists (see config.py).
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

from understudy import config as cfg

# each module imports its heavy dependencies only when run: torch, discord.py, Presidio
COMMANDS = {
    "dataset": ("understudy.dataset", "DiscordChatExporter JSON -> train/val JSONL"),
    "stats": ("understudy.stats", "token lengths of the dataset"),
    "train": ("understudy.train", "QLoRA fine-tune of the shared adapter"),
    "convert": ("understudy.convert", "adapter -> GGUF for llama-server"),
    "serve": ("understudy.serve", "llama-server with the adapter"),
    "bot": ("understudy.bot", "one Discord bot per persona"),
}
# serve and bot run until stopped, so the pipeline ends before them
PIPELINE = ["dataset", "stats", "train", "convert"]


def usage() -> str:
    lines = ["usage: understudy [--config FILE] <command> [args]", "", "commands:"]
    lines += [f"  {name:<9} {summary}" for name, (_, summary) in COMMANDS.items()]
    lines.append(f"  {'pipeline':<9} {' -> '.join(PIPELINE)}, or part of it with --from/--to")
    return "\n".join(lines)


def module(name: str) -> ModuleType:
    return importlib.import_module(COMMANDS[name][0])


def stages(start: str | None, stop: str | None) -> list[str]:
    return PIPELINE[PIPELINE.index(start or PIPELINE[0]):PIPELINE.index(stop or PIPELINE[-1]) + 1]


def pipeline(names: list[str], stage: Callable[[str], int]) -> int:
    for name in names:
        print(f"\n== {name}", flush=True)
        code = stage(name)
        if code:
            print(f"\n{name} failed (exit {code}); stopping", file=sys.stderr)
            return code
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    path = cfg.DEFAULT if cfg.DEFAULT.exists() else None
    if argv[:1] == ["--config"]:
        if len(argv) < 2:
            print("--config needs a FILE", file=sys.stderr)
            return 2
        path, argv = Path(argv[1]), argv[2:]
    if not argv or argv[0] in ("-h", "--help"):
        print(usage())
        return 0 if argv else 2
    name, rest = argv[0], argv[1:]
    if name not in COMMANDS and name != "pipeline":
        print(f"unknown command: {name}\n\n{usage()}", file=sys.stderr)
        return 2

    config: dict = {}
    if path:
        try:
            config = cfg.load(path)
            cfg.check(config, {n: module(n).parser() for n in COMMANDS})
        except cfg.ConfigError as e:
            print(e, file=sys.stderr)
            return 2

    def run(command: str, args: list[str]) -> int:
        mod = module(command)
        parser = mod.parser()
        parser.set_defaults(**cfg.defaults(config, command, parser))
        return mod.run(parser.parse_args(args))

    if name == "pipeline":
        ap = argparse.ArgumentParser(prog="understudy pipeline",
                                     description=f"Runs {' -> '.join(PIPELINE)} with the config's settings.")
        ap.add_argument("--from", dest="start", choices=PIPELINE)
        ap.add_argument("--to", dest="stop", choices=PIPELINE)
        args = ap.parse_args(rest)
        names = stages(args.start, args.stop)
        if not names:
            ap.error("--from comes after --to")
        return pipeline(names, lambda stage: run(stage, []))
    return run(name, rest)
