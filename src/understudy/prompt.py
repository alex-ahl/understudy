"""The prompt for one reply, built the same way for training and for the bots."""

from __future__ import annotations

PROFILE_TAIL = 3    # the persona's own lines go this many turns before the end of the context


def system_prompt(persona: str, channel: str) -> str:
    # training and inference must build this identically, or the adapter sees a stranger
    return f"Reply as `{persona}` in #{channel}."


def build_messages(persona: str, channel: str, turns: list[str], own: list[str]) -> list[dict]:
    """The conversation, with `own` lines as the persona's messages."""
    # the adapter reads the system prompt as a name tag but follows the conversation, and near
    # the end of it; training and the bots both build prompts here, so they can't drift apart
    cut = max(len(turns) - PROFILE_TAIL, 0)
    lines = [*turns[:cut], *(f"{persona}: {line}" for line in own), *turns[cut:]]
    return [
        {"role": "system", "content": system_prompt(persona, channel)},
        {"role": "user", "content": "\n".join(lines)},
    ]
