"""Persona profiles and the system prompt they feed into.

profiles.txt holds one block per persona:

    [bob]
    Two kids. Plays PUBG every night. Can't stand anything Apple makes.

Blank lines and `#` comments are ignored. Names match case-insensitively.
"""

from __future__ import annotations

from pathlib import Path


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


def system_prompt(persona: str, channel: str, profiles: dict[str, str] | None = None) -> str:
    # training and inference must build this identically, or the adapter sees a stranger
    prompt = f"Reply as `{persona}` in #{channel}."
    profile = (profiles or {}).get(persona.lower())
    return f"{prompt}\nAbout {persona}: {' '.join(profile.splitlines())}" if profile else prompt
