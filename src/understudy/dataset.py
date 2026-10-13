"""Turn DiscordChatExporter JSON exports into persona fine-tuning data.

    understudy dataset exports/ --persona bob --persona alice -o data/

Reads DCE JSON, merges message bursts, splits on conversation gaps, windows
context, and writes train.jsonl / val.jsonl in the chat-messages format.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from understudy.profiles import read_profiles, system_prompt

if TYPE_CHECKING:
    from understudy.scrub import Scrubber

# ---------------------------------------------------------------- tunables

BURST_SECONDS = 60       # consecutive messages from one author become one turn
GAP_MINUTES = 10         # a longer silence starts a new conversation
CONTEXT_TURNS = 10       # how many turns of context precede the target
MIN_TARGET_CHARS = 15    # drops "lol", "^", "yeah"
MIN_CONTEXT_TURNS = 2    # the target needs something to be replying to
VAL_FRACTION = 0.08

KEEP_TYPES = {"Default", "Reply"}

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


# ------------------------------------------------------------------ names

def fold_char(ch: str) -> str:
    """Lowercase, accents stripped; stray accent marks like ´ fold to nothing."""
    if unicodedata.category(ch) == "Sk":
        return ""
    base = unicodedata.normalize("NFD", ch)
    return "".join(c for c in base if not unicodedata.combining(c)).lower()


def fold(text: str) -> str:
    return "".join(fold_char(ch) for ch in text)


def unquote(real: str) -> tuple[str, bool]:
    """A quoted alias name ("acme") matches whole words only."""
    if len(real) > 2 and real[0] == real[-1] == '"':
        return real[1:-1], True
    return real, False


def replace_folded(content: str, real: str, fake: str, whole_word: bool = False) -> str:
    """Replace every word containing `real` with `fake`, matching on folded text."""
    folded, origin = [], []
    for i, ch in enumerate(content):
        for f in fold_char(ch):
            folded.append(f)
            origin.append(i)
    folded = "".join(folded)

    spans = [(origin[m.start()], origin[m.end() - 1] + 1)
             for m in re.finditer(rf"(?<!\w){re.escape(real)}s?(?!\w)" if whole_word
                                  else rf"\w*{re.escape(real)}\w*", folded)]
    for start, end in reversed(spans):
        content = content[:start] + fake + content[end:]
    return content


def read_aliases_file(path: Path) -> tuple[list[str], list[str]]:
    """REAL=FAKE lines, and the names marked as personas with a leading *."""
    pairs, marked = [], []
    for line in (l.split("#", 1)[0].strip() for l in path.read_text(encoding="utf-8").splitlines()):
        if line.startswith("*"):
            line = line[1:].strip()
            marked.append(line.partition("=")[2] or line)
            if "=" not in line:
                continue
        if line:
            pairs.append(line)
    return pairs, marked


def build_aliases(pairs: list[str]) -> dict[str, str]:
    """REAL=FAKE pairs keyed by folded real name; raises ValueError on a malformed pair."""
    aliases = {}
    for pair in pairs:
        real, _, fake = pair.partition("=")
        if not real.strip() or not fake.strip():
            raise ValueError(f"alias wants REAL=FAKE, got {pair!r}")
        aliases[fold(real.strip())] = fake.strip()  # quotes survive folding
    return aliases


# ------------------------------------------------------------------ loading

def parse_ts(raw: str) -> datetime:
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def display_name(author: dict, aliases: dict[str, str]) -> str:
    name = (author.get("nickname") or author.get("name") or "unknown").strip()
    key = fold(name)
    return aliases.get(key) or aliases.get(f'"{key}"') or name


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
    # longest first, so "roberta" is replaced before "robert" can claim it
    for key in sorted(aliases, key=lambda k: len(unquote(k)[0]), reverse=True):
        real, whole_word = unquote(key)
        content = replace_folded(content, real, aliases[key], whole_word)

    if scrubber:
        content = scrubber.names(content)

    return content.strip()


def load(paths: list[Path], aliases: dict[str, str], scrubber: Scrubber | None, drop_bots: bool):
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
            if drop_bots and author.get("isBot"):
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
                   by_id: dict[str, Msg], keep_attachment_targets: bool,
                   profiles: dict[str, str]):
    examples = []

    for i, turn in enumerate(turns):
        if fold(turn.author) not in personas:
            continue
        if len(turn.content) < MIN_TARGET_CHARS:
            continue
        if turn.had_attachment and not keep_attachment_targets:
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

        examples.append({"messages": [
            {"role": "system", "content": system_prompt(turn.author, channel, profiles)},
            {"role": "user", "content": "\n".join(lines)},
            {"role": "assistant", "content": turn.content},
        ]})

    return examples


# -------------------------------------------------------------------- main

def optional(name: str) -> Path | None:
    """A file in the working directory, if it's there."""
    return Path(name) if Path(name).exists() else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="understudy dataset", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, nargs="?", default=Path("exports"),
                    help="DCE .json file or a directory of them (default exports/)")
    ap.add_argument("--persona", action="append", default=[],
                    help="name to imitate; repeat for each (matches name or nickname)")
    ap.add_argument("-o", "--out", type=Path, default=Path("data"))
    ap.add_argument("--alias", action="append", default=[], metavar="REAL=FAKE",
                    help="pseudonymize a name everywhere it appears")
    ap.add_argument("--aliases", type=Path, metavar="FILE", default=optional("aliases.txt"),
                    help="REAL=FAKE per line, # comments, a leading * marks a persona "
                         "(*NAME alone for one without an alias), \"REAL\" matches whole words")
    ap.add_argument("--profiles", type=Path, metavar="FILE", default=optional("profiles.txt"),
                    help="persona profiles to put in each system prompt (see profiles.py)")
    ap.add_argument("--no-scrub", action="store_true",
                    help="keep links, emails, phone numbers, addresses, towns, passwords and names verbatim")
    ap.add_argument("--keep-bots", action="store_true")
    ap.add_argument("--keep-attachment-targets", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    paths = sorted(args.input.rglob("*.json")) if args.input.is_dir() else [args.input]
    if not paths:
        print(f"no .json exports under {args.input}", file=sys.stderr)
        return 1

    pairs = list(args.alias)
    marked = []
    if args.aliases:
        file_pairs, marked = read_aliases_file(args.aliases)
        pairs += file_pairs

    try:
        aliases = build_aliases(pairs)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1

    personas = {fold(unquote(p.strip())[0]) for p in args.persona + marked}
    if not personas:
        print("no personas: pass --persona or mark lines in --aliases with *", file=sys.stderr)
        return 1
    personas = {fold(aliases.get(p, p)) for p in personas}

    profiles = read_profiles(args.profiles)
    scrubber = None
    if not args.no_scrub:
        from understudy.scrub import Scrubber  # loads Presidio, spaCy and GLiNER
        scrubber = Scrubber(allow={fold(fake) for fake in aliases.values()} | personas)
    msgs, skipped = load(paths, aliases, scrubber, not args.keep_bots)
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
        examples += build_examples(turns, personas, channel, by_id,
                                   args.keep_attachment_targets, profiles)

    if not examples:
        seen = Counter(m.author for m in msgs).most_common(15)
        print("no examples built. authors present:", file=sys.stderr)
        for name, n in seen:
            print(f"  {name}  ({n})", file=sys.stderr)
        return 1

    random.Random(args.seed).shuffle(examples)
    cut = max(1, int(len(examples) * VAL_FRACTION))
    val, train = examples[:cut], examples[cut:]

    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        with (args.out / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    # keyed by name alone: the system prompt also carries the channel
    per_persona = Counter(e["messages"][0]["content"].split("`")[1] for e in examples)
    print(f"loaded     {len(msgs)} messages from {len(paths)} export(s)")
    for reason, n in skipped.most_common():
        print(f"  skipped  {n} ({reason})")
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


if __name__ == "__main__":
    raise SystemExit(main())
