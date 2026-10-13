"""Persona profiles, and the prompt they go into.

profiles.txt holds one block per persona:

    [bob]
    Two kids. Plays PUBG every night. Can't stand anything Apple makes.

Blank lines and `#` comments are ignored. Names match case-insensitively.
"""

from __future__ import annotations

from pathlib import Path

PROFILE_TAIL = 3    # the persona's own lines go this many turns before the end of the context


def read_profiles(path: Path | None) -> dict[str, str]:
    if not path or not path.exists():
        return {}
    profiles: dict[str, list[str]] = {}
    current = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1].strip().lower()
            profiles.setdefault(current, [])
        elif line and current:
            profiles[current].append(line)
    return {name: "\n".join(lines) for name, lines in profiles.items() if lines}


def system_prompt(persona: str, channel: str) -> str:
    # training and inference must build this identically, or the adapter sees a stranger
    return f"Reply as `{persona}` in #{channel}."


def build_messages(persona: str, channel: str, turns: list[str], own: list[str]) -> list[dict]:
    """The prompt for one reply: the conversation, with `own` lines as the persona's messages."""
    # the adapter reads the system prompt as a name tag but follows the conversation, and near
    # the end of it; training and the bots both build prompts here, so they can't drift apart
    cut = max(len(turns) - PROFILE_TAIL, 0)
    lines = [*turns[:cut], *(f"{persona}: {line}" for line in own), *turns[cut:]]
    return [
        {"role": "system", "content": system_prompt(persona, channel)},
        {"role": "user", "content": "\n".join(lines)},
    ]
