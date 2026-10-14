"""Turn DiscordChatExporter JSON exports into persona fine-tuning data.

    understudy dataset private/exports/ --persona bob --persona alice -o private/data/

Reads DCE JSON, merges message bursts, splits on conversation gaps, windows
context, and writes train.jsonl / val.jsonl in the chat-messages format.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from understudy.people import fold, read as read_people, rename
from understudy.prompt import build_messages

if TYPE_CHECKING:
    from understudy.scrub import Scrubber

# ---------------------------------------------------------------- tunables

BURST_SECONDS = 60       # consecutive messages from one author become one turn
GAP_MINUTES = 10         # a longer silence starts a new conversation
CONTEXT_TURNS = 10       # how many turns of context precede the target
MIN_TARGET_CHARS = 15    # drops "lol", "^", "yeah"
MIN_CONTEXT_TURNS = 2    # the target needs something to be replying to
VAL_FRACTION = 0.08
KEEP = 0.2               # share of --downweight matches kept, in replies and in context
VOICE_EXAMPLES = 3       # most real one-liners put in a context; the bots always use this many
VOICE_CHARS = (12, 90)

KEEP_TYPES = {"Default", "Reply"}

RE_USER_MENTION = re.compile(r"<@!?(\d+)>")
RE_ROLE_MENTION = re.compile(r"<@&\d+>")
RE_CHAN_MENTION = re.compile(r"<#\d+>")
RE_LINK = re.compile(r"\[link [^\]]*\]|https?://\S+|\bwww\.\S+")


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
class Example:
    persona: str
    channel: str
    lines: list[str]
    reply: str


@dataclass
class Turn:
    author: str
    ts: datetime
    content: str
    ids: list[str] = field(default_factory=list)
    had_attachment: bool = False


# ------------------------------------------------------------------ loading

def parse_ts(raw: str) -> datetime:
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def display_name(author: dict, aliases: dict[str, str]) -> str:
    name = (author.get("nickname") or author.get("name") or "unknown").strip()
    return aliases.get(fold(name)) or name


def clean(content: str, mentions: list[dict], aliases: dict[str, str],
          scrubber: Scrubber | None = None) -> str:
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


def load(paths: list[Path], aliases: dict[str, str], scrubber: Scrubber | None):
    msgs: list[Msg] = []
    skipped = Counter()

    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        channel = (data.get("channel") or {}).get("name") or path.stem

        for m in data.get("messages", []):
            if m.get("type") not in KEEP_TYPES:
                skipped["system message"] += 1
                continue

            author = m.get("author") or {}
            if author.get("isBot"):
                skipped["bot"] += 1
                continue

            content = clean(m.get("content", ""), m.get("mentions") or [], aliases, scrubber)
            if not content:
                skipped["empty / attachment-only"] += 1
                continue

            msgs.append(Msg(
                id=m.get("id", ""),
                ts=parse_ts(m["timestamp"]),
                author=display_name(author, aliases),
                content=content,
                channel=channel,
                reply_to=(m.get("reference") or {}).get("messageId"),
                had_attachment=bool(m.get("attachments")),
            ))

    return msgs, skipped


# ------------------------------------------------------------ burst merging

def merge_bursts(msgs: list[Msg]) -> list[Turn]:
    """Three messages fired off in a row are one conversational turn."""
    turns: list[Turn] = []
    for m in msgs:
        last = turns[-1] if turns else None
        if (last and last.author == m.author
                and (m.ts - last.ts) <= timedelta(seconds=BURST_SECONDS)):
            last.content += "\n" + m.content
            last.ts = m.ts
            last.ids.append(m.id)
            last.had_attachment |= m.had_attachment
        else:
            turns.append(Turn(m.author, m.ts, m.content, [m.id], m.had_attachment))
    return turns


# --------------------------------------------------------------- windowing

def render(turn: Turn) -> str:
    body = turn.content
    if turn.had_attachment:
        body += " [attachment]"
    return f"{turn.author}: {body}"


def build_examples(turns: list[Turn], personas: set[str], channel: str,
                   by_id: dict[str, Msg]) -> list[Example]:
    examples = []

    for i, turn in enumerate(turns):
        if fold(turn.author) not in personas:
            continue
        if len(turn.content) < MIN_TARGET_CHARS:
            continue
        if turn.had_attachment:
            continue  # the message meant something only alongside its image

        # walk back until the conversation breaks
        ctx: list[Turn] = []
        cursor = turn.ts
        for prev in reversed(turns[max(0, i - CONTEXT_TURNS):i]):
            if (cursor - prev.ts) > timedelta(minutes=GAP_MINUTES):
                break
            ctx.append(prev)
            cursor = prev.ts
        ctx.reverse()

        if len(ctx) < MIN_CONTEXT_TURNS:
            continue

        lines = [render(t) for t in ctx]

        # an explicit reply to something that scrolled out of the window
        if turn.ids:
            ref = by_id.get(turn.ids[0], None)
            ref_id = ref.reply_to if ref else None
            if ref_id and not any(ref_id in t.ids for t in ctx):
                quoted = by_id.get(ref_id)
                if quoted:
                    lines.insert(0, f"> replying to {quoted.author}: {quoted.content}")

        examples.append(Example(turn.author, channel, lines, turn.content))

    return examples


# ----------------------------------------------------------------- shaping

def has_link(text: str) -> bool:
    return bool(RE_LINK.search(text))


def any_of(patterns: list[str]) -> re.Pattern | None:
    """One case-insensitive regex matching any of the patterns."""
    return re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE) if patterns else None


def downweight(examples: list[Example], pattern: re.Pattern, rng: random.Random) -> list[Example]:
    """Matching replies and context lines, kept at KEEP; the line replied to always stays.

    A topic the context is full of is what every next reply continues, so thinning it in the
    context too is what stops the model from chaining it."""
    kept = []
    for e in examples:
        if pattern.search(e.reply) and rng.random() >= KEEP:
            continue
        *earlier, last = e.lines
        earlier = [l for l in earlier if not pattern.search(l) or rng.random() < KEEP]
        kept.append(Example(e.persona, e.channel, [*earlier, last], e.reply))
    return kept


def cap(examples: list[Example], n: int, rng: random.Random) -> list[Example]:
    """At most n examples per persona, sampled."""
    by_persona: dict[str, list[Example]] = {}
    for e in examples:
        by_persona.setdefault(e.persona, []).append(e)
    return [e for group in by_persona.values() for e in (rng.sample(group, n) if len(group) > n else group)]


def voice_pool(examples: list[Example], pattern: re.Pattern | None) -> dict[str, list[str]]:
    """Per persona (folded), plain one-liners that show how they write and pull nowhere in particular."""
    pool: dict[str, list[str]] = {}
    lo, hi = VOICE_CHARS
    for e in examples:
        r = e.reply
        if ("\n" not in r and lo <= len(r) <= hi and "[" not in r and not has_link(r)
                and not (pattern and pattern.search(r))):
            pool.setdefault(fold(e.persona), []).append(r)
    return pool


def own_lines(reply: str, pool: list[str], profile: list[str], rng: random.Random) -> list[str]:
    """Up to VOICE_EXAMPLES voice lines, then some of the profile in order; sometimes nothing.

    Varied per example: text that is the same in every example teaches nothing."""
    others = [l for l in pool if l != reply]
    voices = rng.sample(others, min(rng.randint(0, VOICE_EXAMPLES), len(others)))
    picked = sorted(rng.sample(range(len(profile)), rng.randint(0, len(profile))))
    return voices + [profile[i] for i in picked]


# -------------------------------------------------------------------- main

def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="understudy dataset", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, nargs="?", default=Path("private/exports"),
                    help="DCE .json file or a directory of them (default private/exports/)")
    ap.add_argument("--persona", action="append", default=[],
                    help="name to imitate, on top of the config's [persona.NAME] tables; repeatable")
    ap.add_argument("-o", "--out", type=Path, default=Path("private/data"))
    ap.add_argument("--alias", action="append", default=[], metavar="REAL=FAKE",
                    help="pseudonymize a name everywhere it appears, on top of the config's [aliases]")
    ap.add_argument("--downweight", action="append", default=[], metavar="REGEX",
                    help=f"matching replies and context lines are kept at {KEEP * 100:.0f}%%; repeatable")
    ap.add_argument("--max-per-persona", type=int, metavar="N", help="sample each persona down to N examples")
    ap.add_argument("--no-scrub", action="store_true",
                    help="keep links, emails, phone numbers, addresses, towns, passwords and names verbatim")
    ap.add_argument("--seed", type=int, default=0)
    ap.set_defaults(people={})  # the config's [persona.*] and [aliases] tables
    return ap


def run(args: argparse.Namespace) -> int:
    paths = sorted(args.input.rglob("*.json")) if args.input.is_dir() else [args.input]
    if not paths:
        print(f"no .json exports under {args.input}", file=sys.stderr)
        return 1

    try:
        people = read_people(args.people, args.persona, args.alias)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    if not people.personas:
        print("no personas: add [persona.NAME] tables to the config or pass --persona", file=sys.stderr)
        return 1
    aliases = people.aliases
    personas = {fold(p.name) for p in people.personas}
    profiles = {fold(p.name): p.profile for p in people.personas}

    scrubber = None
    if not args.no_scrub:
        from understudy.scrub import Scrubber  # loads Presidio, spaCy and GLiNER
        scrubber = Scrubber(allow={fold(fake) for fake in aliases.values()} | personas)
    msgs, skipped = load(paths, aliases, scrubber)
    if not msgs:
        print("nothing survived loading", file=sys.stderr)
        return 1

    by_id = {m.id: m for m in msgs}

    by_channel: dict[str, list[Msg]] = {}
    for m in msgs:
        by_channel.setdefault(m.channel, []).append(m)

    examples = []
    for channel, channel_msgs in by_channel.items():
        channel_msgs.sort(key=lambda m: m.ts)
        turns = merge_bursts(channel_msgs)
        examples += build_examples(turns, personas, channel, by_id)

    if not examples:
        seen = Counter(m.author for m in msgs).most_common(15)
        print("no examples built. authors present:", file=sys.stderr)
        for name, n in seen:
            print(f"  {name}  ({n})", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    built = len(examples)
    examples = [e for e in examples if not has_link(e.reply)]
    dropped = {"link": built - len(examples)}
    pattern = any_of(args.downweight)
    if pattern:
        n = len(examples)
        examples = downweight(examples, pattern, rng)
        dropped["downweighted"] = n - len(examples)
    if args.max_per_persona:
        n = len(examples)
        examples = cap(examples, args.max_per_persona, rng)
        dropped["over --max-per-persona"] = n - len(examples)

    rng.shuffle(examples)
    cut = max(1, int(len(examples) * VAL_FRACTION))
    val, train = examples[:cut], examples[cut:]
    # from train only, so no val reply shows up in a training context
    voices = voice_pool(train, pattern)

    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        with (args.out / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for e in rows:
                own = own_lines(e.reply, voices.get(fold(e.persona), []),
                                profiles.get(fold(e.persona), []), rng)
                messages = build_messages(e.persona, e.channel, e.lines, own)
                messages.append({"role": "assistant", "content": e.reply})
                fh.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
    with (args.out / "voices.json").open("w", encoding="utf-8") as fh:
        json.dump(voices, fh, ensure_ascii=False, indent=1)

    per_persona = Counter(e.persona for e in examples)
    print(f"loaded     {len(msgs)} messages from {len(paths)} export(s)")
    for reason, n in skipped.most_common():
        print(f"  skipped  {n} ({reason})")
    for reason, n in dropped.items():
        if n:
            print(f"  dropped  {n} ({reason})")
    print(f"wrote      {len(train)} train / {len(val)} val -> {args.out}")
    print("examples per persona:")
    for name, n in per_persona.most_common():
        print(f"  {n:>6}  {name}")
    thin = [name for name, n in per_persona.items() if n < 500]
    if thin:
        print("\nunder ~500 examples is thin for a convincing persona:")
        for name in thin:
            print(f"  {name}")
    if scrubber and scrubber.organizations:
        print("\norganisations mentioned, kept as is; alias the ones that identify someone:")
        for name, n in scrubber.organizations.most_common(20):
            print(f"  {n:>6}  {name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
