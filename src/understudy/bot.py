"""Discord bots that let the personas talk in the server, through llama-server.

    understudy bot

Each persona with a `token` in its [persona.NAME] table is its own bot account.
The voice lines come from private/data/voices.json, written by understudy
dataset. A persona replies to the channel's recent conversation when someone
@-mentions it or writes its name in a message. With --converse-every, the
personas also talk among themselves in --converse-channel: one opens by turning
to another, whoever is named answers next, and a person writing stops them.

--llm-url and --channel default to the LLM_URL and CHANNELS (comma-separated)
environment variables.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from understudy.chat import (
    CONTEXT_TURNS,
    GAP_MINUTES,
    VOICE_EXAMPLES,
    Msg,
    any_of,
    clean,
    display_name,
    merge_bursts,
    render,
)
from understudy.people import People, fold
from understudy.people import read as read_people
from understudy.prompt import build_messages

log = logging.getLogger("understudy")

HISTORY = 40  # raw messages; bursts and the gap cut bring it down to a conversation
MAX_LINES = 5  # real bursts rarely run longer; loops always do

RE_PLACEHOLDER = re.compile(r"\[(?:link [^\]]*|[a-z]+)\]")
REROLLS = 2  # extra tries for a reply that matches --downweight
REPLY_TIMEOUT = 300  # seconds; a CPU server takes well under a minute per reply


def named_personas(text: str, personas: list[str]) -> list[str]:
    """Personas whose pseudonym appears as a word, in the order they appear."""
    folded = fold(text)
    hits = []
    for persona in personas:
        m = re.search(rf"(?<!\w){re.escape(fold(persona))}(?!\w)", folded)
        if m:
            hits.append((m.start(), persona))
    return [p for _, p in sorted(hits)]


def conversation(msgs: list[Msg]) -> list[str]:
    """Recent messages as training-style turns, cut at the last long silence."""
    turns = merge_bursts(msgs)
    kept = []
    for turn in reversed(turns):
        if kept and (kept[-1].ts - turn.ts).total_seconds() > GAP_MINUTES * 60:
            break
        kept.append(turn)
    return [render(t) for t in reversed(kept)]


def next_speaker(reply: str, current: str, personas: list[str], rng: random.Random) -> str:
    """Whoever the reply names, else another persona at random."""
    others = [p for p in personas if p != current]
    named = named_personas(reply, others)
    return named[0] if named else rng.choice(others)


def own_lines(persona: str, voices: dict[str, list[str]], profile: list[str], rng: random.Random) -> list[str]:
    """Real one-liners, then the whole profile: the most of what training put in a context."""
    pool = voices.get(fold(persona), [])
    return [*rng.sample(pool, min(VOICE_EXAMPLES, len(pool))), *profile]


def server_reply(url: str, messages: list[dict], sampling: dict[str, float], prefill: str = "") -> str:
    """Any OpenAI-compatible endpoint: llama-server, Ollama, LM Studio.

    sampling holds the request's sampler fields, like temperature. A prefill is the start
    of the reply, written for the model to continue."""
    if prefill:
        messages = [*messages, {"role": "assistant", "content": prefill}]
    # dry_* is llama-server's repeat-from-context penalty; without it a persona echoes its
    # own previous line verbatim. Other servers ignore the extra fields
    body = json.dumps(
        {
            "model": "understudy",
            "messages": messages,
            "max_tokens": 200,
            **sampling,
            "dry_multiplier": 0.8,
            "dry_base": 1.75,
            "dry_allowed_length": 2,
        }
    ).encode()
    req = urllib.request.Request(f"{url.rstrip('/')}/v1/chat/completions", body, {"Content-Type": "application/json"})
    # without a timeout, a stalled server holds the channel's lock and every later reply
    with urllib.request.urlopen(req, timeout=REPLY_TIMEOUT) as resp:
        return with_prefill(json.load(resp)["choices"][0]["message"]["content"].strip(), prefill)


def server_ready(url: str) -> bool:
    """Whether llama-server has loaded its model; it answers 503 until then."""
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/health", timeout=5) as resp:
            return resp.status == 200
    except OSError:
        return False


def with_prefill(text: str, prefill: str) -> str:
    # llama-server returns the prefill with its continuation; some servers return only the rest
    return text if not prefill or text.startswith(prefill.strip()) else f"{prefill.strip()} {text}"


def clean_reply(reply: str, persona: str, speakers: set[str]) -> str:
    """The persona's own lines, cut where the model starts writing for someone else or looping,
    with scrub placeholders taken out."""
    kept: list[str] = []
    for line in reply.splitlines():
        speaker, sep, rest = line.partition(": ")
        if sep and fold(speaker.strip()) == fold(persona):
            line = rest
        elif sep and fold(speaker.strip()) in speakers:
            break
        line = re.sub(r"[ \t]{2,}", " ", RE_PLACEHOLDER.sub("", line)).strip()
        if not line:
            continue
        if line in kept or len(kept) == MAX_LINES:
            break
        kept.append(line)
    return "\n".join(kept)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="understudy bot", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--voices", type=Path, default=Path("private/data/voices.json"), metavar="FILE")
    ap.add_argument(
        "--llm-url",
        default=os.environ.get("LLM_URL", "http://localhost:8080"),
        help="OpenAI-compatible server (default http://localhost:8080)",
    )
    ap.add_argument(
        "--channel",
        action="append",
        metavar="NAME",
        default=[c.strip() for c in os.environ.get("CHANNELS", "").split(",") if c.strip()],
        help="channel to reply in; repeat for each (default: all)",
    )
    ap.add_argument(
        "--converse-every",
        type=float,
        default=0,
        metavar="MINUTES",
        help="start a conversation among the personas this often (default 0: never)",
    )
    ap.add_argument("--converse-length", type=int, default=20, metavar="N", help="messages per conversation")
    ap.add_argument("--converse-channel", metavar="NAME", help="channel the conversations happen in")
    ap.add_argument(
        "--downweight",
        action="append",
        default=[],
        metavar="REGEX",
        help=f"a matching reply is generated again, up to {REROLLS} times, then dropped",
    )
    ap.add_argument("--temperature", type=float, default=1.0, help="sampling temperature (default 1.0)")
    ap.add_argument(
        "--min-p",
        type=float,
        default=0.05,
        help="drop tokens less likely than this share of the likeliest (default 0.05)",
    )
    ap.add_argument(
        "--xtc-probability",
        type=float,
        default=0.5,
        help="how often XTC cuts the likeliest tokens; 0 turns it off (default 0.5)",
    )
    ap.add_argument(
        "--xtc-threshold",
        type=float,
        default=0.1,
        help="tokens at least this likely are the ones XTC can cut (default 0.1)",
    )
    ap.set_defaults(people={})  # the config's [persona.*] and [aliases] tables
    return ap


class Bots:
    """One Discord client per persona, sharing what they have seen."""

    def __init__(
        self, people: People, voices: dict[str, list[str]], args: argparse.Namespace, rng: random.Random | None = None
    ) -> None:
        self.aliases = people.aliases
        self.tokens = {p.name: p.token for p in people.personas if p.token}
        self.profiles = {p.name: p.profile for p in people.personas}
        self.personas = list(self.tokens)
        self.voices = voices
        self.llm_url: str = args.llm_url
        self.channels = set(args.channel)
        self.avoid = any_of(args.downweight)
        self.sampling: dict[str, float] = {
            "temperature": args.temperature,
            "min_p": args.min_p,
            "xtc_probability": args.xtc_probability,
            "xtc_threshold": args.xtc_threshold,
        }
        self.converse_length: int = args.converse_length
        self.rng = rng or random.Random()
        self.clients: dict[str, Any] = {}
        self.persona_of: dict[int, str] = {}  # bot user id -> persona, filled as each logs in
        self.locks: dict[int, asyncio.Lock] = {}
        self.last_person: dict[int, datetime] = {}  # channel id -> when a person last wrote there

    def person_name(self, user) -> str:
        # the account name, the same everywhere; server nicknames can be aliased to a persona
        return display_name(
            {"name": user.name, "nickname": getattr(user, "global_name", None) or user.name}, self.aliases
        )

    async def history(self, channel, since: datetime | None = None) -> list[str]:
        msgs = []
        async for m in channel.history(limit=HISTORY, after=since, oldest_first=False):
            if m.author.bot and m.author.id not in self.persona_of:
                continue  # other bots, like the training data
            mentions = [
                {"id": str(u.id), "name": u.name, "nickname": self.persona_of.get(u.id) or u.display_name}
                for u in m.mentions
            ]
            text = clean(m.content, mentions, self.aliases)
            if not text and not m.attachments:
                continue
            author = self.persona_of.get(m.author.id) or self.person_name(m.author)
            msgs.append(
                Msg(
                    id=str(m.id),
                    ts=m.created_at,
                    author=author,
                    content=text,
                    channel=channel.name,
                    reply_to=None,
                    had_attachment=bool(m.attachments),
                )
            )
        return conversation(list(reversed(msgs)))

    async def speak(self, channel, persona: str, prefill: str = "", since: datetime | None = None) -> str:
        import discord

        async with self.locks.setdefault(channel.id, asyncio.Lock()):
            async with channel.typing():
                turns = await self.history(channel, since)
                own = own_lines(persona, self.voices, self.profiles[persona], self.rng)
                messages = build_messages(persona, turns[-CONTEXT_TURNS:], own)
                speakers = {fold(p) for p in self.personas} | {fold(t.split(": ", 1)[0]) for t in turns}
                for _ in range(1 + REROLLS):
                    raw = await asyncio.to_thread(server_reply, self.llm_url, messages, self.sampling, prefill)
                    # matched on the raw reply: cleanup removes some of what the patterns look for
                    if not (self.avoid and self.avoid.search(raw)):
                        break
                    log.info("%s's reply matched --downweight, generating again", persona)
                else:
                    return ""
            text = clean_reply(raw, persona, speakers)
            if text:
                await channel.send(text, allowed_mentions=discord.AllowedMentions.none())
                log.info("%s replied in #%s", persona, channel.name)
            return text

    async def converse(self, channel_id: int) -> None:
        started = datetime.now(UTC)
        persona = self.rng.choice(self.personas)
        # opens by turning to someone, so it starts its own thread instead of carrying on the last one
        prefill = f"@{next_speaker('', persona, self.personas, self.rng)} "
        for _ in range(self.converse_length):
            if self.last_person.get(channel_id, started) > started:
                log.info("conversation stops: a person wrote")
                return
            # only what was said since it started: each conversation begins blank
            text = await self.speak(self.clients[persona].get_channel(channel_id), persona, prefill, started)
            if not text:
                log.info("conversation stops: %s had nothing to say", persona)
                return
            prefill = ""
            persona = next_speaker(text, persona, self.personas, self.rng)
            await asyncio.sleep(self.rng.uniform(3, 8))  # a beat between people, not a script

    async def converse_every(self, minutes: float, name: str) -> None:
        import discord

        first = self.clients[self.personas[0]]
        # wait_until_ready() raises before a client has logged in, and the logins are staggered
        while not all(c.is_ready() for c in self.clients.values()) or not await asyncio.to_thread(
            server_ready, self.llm_url
        ):
            await asyncio.sleep(5)
        channel = discord.utils.get(first.get_all_channels(), name=name, type=discord.ChannelType.text)
        if channel is None:
            log.warning("no text channel named %s; no conversations", name)
            return
        while True:
            # runs under gather with the bots: an error escaping here would stop all of them
            try:
                await self.converse(channel.id)
            except Exception:
                log.exception("conversation failed")
            await asyncio.sleep(minutes * 60)

    def wire(self, persona: str, client) -> None:
        import discord

        @client.event
        async def on_ready():
            self.persona_of[client.user.id] = persona
            log.info("%s logged in as %s", persona, client.user)

        @client.event
        async def on_message(message):
            # only people trigger replies, or the personas would answer each other forever
            if message.author.bot or not isinstance(message.channel, discord.TextChannel):
                return
            self.last_person[message.channel.id] = message.created_at
            if self.channels and message.channel.name not in self.channels:
                return
            # typing @name often autocompletes to the role Discord creates for the bot
            mentioned = any(u.id == client.user.id for u in message.mentions) or any(
                r.tags and r.tags.bot_id == client.user.id for r in message.role_mentions
            )
            if mentioned or persona in named_personas(message.content, self.personas):
                # each bot posts through its own connection, so the channel comes from this client
                await self.speak(client.get_channel(message.channel.id), persona)

    async def start(self, persona: str, delay: float) -> None:
        # a small gap so the logins don't all hit Discord in the same instant
        await asyncio.sleep(delay)
        # one bad token or missing intent shouldn't take the other personas down with it
        try:
            await self.clients[persona].start(self.tokens[persona])
        except Exception as e:  # noqa: BLE001
            log.error("%s could not connect: %s: %s", persona, type(e).__name__, e)

    async def run(self, converse_every: float, converse_channel: str | None) -> None:
        import discord

        # client.run() sets this up itself; start() doesn't, and connection trouble goes silent
        discord.utils.setup_logging()
        intents = discord.Intents.default()
        intents.message_content = True
        self.clients = {p: discord.Client(intents=intents) for p in self.personas}
        for persona, client in self.clients.items():
            self.wire(persona, client)
        tasks = [self.start(p, i) for i, p in enumerate(self.personas)]
        if converse_every and converse_channel:
            tasks.append(self.converse_every(converse_every, converse_channel))
        await asyncio.gather(*tasks)


def run(args: argparse.Namespace) -> int:
    try:
        people = read_people(args.people)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    voices = json.loads(args.voices.read_text(encoding="utf-8")) if args.voices.exists() else {}
    bots = Bots(people, voices, args)
    if not bots.personas:
        print('no persona has a token: add token = "..." to its [persona.NAME] table', file=sys.stderr)
        return 1
    if args.converse_every and not args.converse_channel:
        print("--converse-every needs --converse-channel", file=sys.stderr)
        return 1
    if args.converse_every and len(bots.personas) < 2:
        print("conversations need at least two personas with a token", file=sys.stderr)
        return 1
    asyncio.run(bots.run(args.converse_every, args.converse_channel))
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
