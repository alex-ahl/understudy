"""What a conversation is, for training and the bots alike: messages, turns, and how
they are cleaned and rendered. Both must agree, or the adapter reads a stranger's chat.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from understudy.people import fold, rename

if TYPE_CHECKING:
    from understudy.scrub import Scrubber

BURST_SECONDS = 60  # consecutive messages from one author become one turn
GAP_MINUTES = 10  # a longer silence starts a new conversation
CONTEXT_TURNS = 10  # how many turns of context precede the target
VOICE_EXAMPLES = 3  # most real one-liners put in a context; the bots always use this many

RE_USER_MENTION = re.compile(r"<@!?(\d+)>")
RE_ROLE_MENTION = re.compile(r"<@&\d+>")
RE_CHAN_MENTION = re.compile(r"<#\d+>")


@dataclass
class Msg:
    id: str
    ts: datetime
    author: str
    content: str
    channel: str
    reply_to: str | None
    had_attachment: bool


@dataclass
class Turn:
    author: str
    ts: datetime
    content: str
    ids: list[str] = field(default_factory=list)
    had_attachment: bool = False


def display_name(author: dict, aliases: dict[str, str]) -> str:
    name = (author.get("nickname") or author.get("name") or "unknown").strip()
    return aliases.get(fold(name)) or name


def clean(content: str, mentions: list[dict], aliases: dict[str, str], scrubber: Scrubber | None = None) -> str:
    by_id = {m.get("id"): display_name(m, aliases) for m in mentions}
    content = RE_USER_MENTION.sub(lambda m: "@" + by_id.get(m.group(1), "someone"), content)
    content = RE_ROLE_MENTION.sub("@role", content)
    content = RE_CHAN_MENTION.sub("#channel", content)

    # before aliasing, which would rewrite names inside links and streets and strand the rest
    if scrubber:
        content = scrubber.contact(content)

    # alias anyone named in plain text too, so pseudonyms stay consistent.
    # DCE already renders mentions as "@nickname", so those land here, not in RE_USER_MENTION
    content = rename(content, aliases)

    if scrubber:
        content = scrubber.names(content)

    return content.strip()


def merge_bursts(msgs: list[Msg]) -> list[Turn]:
    """Three messages fired off in a row are one conversational turn."""
    turns: list[Turn] = []
    for m in msgs:
        last = turns[-1] if turns else None
        if last and last.author == m.author and (m.ts - last.ts) <= timedelta(seconds=BURST_SECONDS):
            last.content += "\n" + m.content
            last.ts = m.ts
            last.ids.append(m.id)
            last.had_attachment |= m.had_attachment
        else:
            turns.append(Turn(m.author, m.ts, m.content, [m.id], m.had_attachment))
    return turns


def render(turn: Turn) -> str:
    body = turn.content
    if turn.had_attachment:
        body += " [attachment]"
    return f"{turn.author}: {body}"


def any_of(patterns: list[str]) -> re.Pattern | None:
    """One case-insensitive regex matching any of the patterns."""
    return re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE) if patterns else None
