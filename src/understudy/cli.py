"""understudy <command> [args]: each command takes its module's own flags (-h)."""

from __future__ import annotations

import importlib
import sys

# imported only when run: train and stats pull in torch, bot pulls in discord.py
COMMANDS = {
    "dataset": ("understudy.dataset", "DiscordChatExporter JSON -> train/val JSONL"),
    "stats": ("understudy.stats", "token lengths of the dataset"),
    "train": ("understudy.train", "QLoRA fine-tune of the shared adapter"),
    "bot": ("understudy.bot", "one Discord bot per persona"),
}


def usage() -> str:
    lines = ["usage: understudy <command> [args]", "", "commands:"]
    lines += [f"  {name:<9} {summary}" for name, (_, summary) in COMMANDS.items()]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(usage())
        return 0 if argv else 2
    name, rest = argv[0], argv[1:]
    if name not in COMMANDS:
        print(f"unknown command: {name}\n\n{usage()}", file=sys.stderr)
        return 2
    module = importlib.import_module(COMMANDS[name][0])
    return module.main(rest)
