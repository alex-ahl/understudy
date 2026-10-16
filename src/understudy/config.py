"""private/understudy.toml: people and settings for every command, kept out of git.

    [persona.bob]                   # who to imitate (see people.py)
    names = ["robert_1987"]

    [aliases]                       # who else to rename
    carl_xx = "carl"

    max_seq_length = 1536           # top level: every command with that option

    [train]                         # a section: that command only
    epochs = 3

Option keys are the long option names with `_` for `-`. A [command] key wins
over a top-level one, and flags on the command line win over both. Paths are
relative to the working directory, like the defaults.
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

DEFAULT = Path("private/understudy.toml")
PEOPLE = ("persona", "aliases")
PERSONA_KEYS = {"names": list, "profile": list, "token": str}


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
    check_personas(config.get("persona", {}))
    check_aliases(config.get("aliases", {}))
    check_options(config, parsers)


def check_personas(personas: dict) -> None:
    for name, table in personas.items():
        if not isinstance(table, dict):
            raise ConfigError(f"persona.{name} must be a table: [persona.{name}]")
        for key, value in table.items():
            if key not in PERSONA_KEYS:
                raise ConfigError(f"[persona.{name}] has no key {key}; keys are {', '.join(PERSONA_KEYS)}")
            if not isinstance(value, PERSONA_KEYS[key]):
                raise ConfigError(f"[persona.{name}] {key} must be a {PERSONA_KEYS[key].__name__}")


def check_aliases(aliases: dict) -> None:
    for real, fake in aliases.items():
        if not isinstance(fake, str):
            raise ConfigError(f"[aliases] {real} must be a string")


def check_options(config: dict, parsers: dict[str, argparse.ArgumentParser]) -> None:
    for key, value in config.items():
        if key in PEOPLE:
            continue
        if isinstance(value, dict):
            if key not in parsers:
                raise ConfigError(f"unknown section [{key}]; sections are {', '.join([*PEOPLE, *parsers])}")
            unknown = sorted(set(value) - set(options(parsers[key])))
            if unknown:
                raise ConfigError(f"[{key}] has no option {', '.join(unknown)}")
        elif not any(key in options(p) for p in parsers.values()):
            raise ConfigError(f"no command has an option {key}")


def defaults(config: dict, command: str, parser: argparse.ArgumentParser) -> dict:
    """The config's values for one command, as argparse defaults."""
    known = options(parser)
    shared = {k: v for k, v in config.items() if not isinstance(v, dict) and k in known}
    values = shared | config.get(command, {})
    if parser.get_default("people") is not None:
        values["people"] = {k: config[k] for k in PEOPLE if k in config}
    return values
