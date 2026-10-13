"""understudy.toml: settings for every command, kept out of git.

    model = "AI-Sweden-Models/Llama-3-8B-instruct"
    max_seq_length = 1536

    [train]
    epochs = 3

Keys are the long option names with `_` for `-`. A top-level key goes to every
command that has that option; a key in a [command] section goes to that command
only and wins over the top level. Flags on the command line win over both.
Relative paths resolve against the directory holding the config file.
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

DEFAULT = Path("understudy.toml")


class ConfigError(Exception):
    pass


def load(path: Path) -> dict:
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise ConfigError(f"{path}: {e}") from None


def options(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    return {a.dest: a for a in parser._actions if a.dest != "help"}


def check(config: dict, parsers: dict[str, argparse.ArgumentParser]) -> None:
    """Raises ConfigError on a section or key no command has, so a typo can't fall back to a default."""
    for key, value in config.items():
        if isinstance(value, dict):
            if key not in parsers:
                raise ConfigError(f"unknown section [{key}]; commands are {', '.join(parsers)}")
            unknown = sorted(set(value) - set(options(parsers[key])))
            if unknown:
                raise ConfigError(f"[{key}] has no option {', '.join(unknown)}")
        elif not any(key in options(p) for p in parsers.values()):
            raise ConfigError(f"no command has an option {key}")


def defaults(config: dict, command: str, parser: argparse.ArgumentParser, base: Path) -> dict:
    """The config's values for one command, as argparse defaults."""
    known = options(parser)
    shared = {k: v for k, v in config.items() if not isinstance(v, dict) and k in known}
    values = shared | config.get(command, {})
    return {k: resolve(v, known[k], base) for k, v in values.items()}


def resolve(value, action: argparse.Action, base: Path):
    if action.type is not Path:
        return value
    return [base / v for v in value] if isinstance(value, list) else base / value
