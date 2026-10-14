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
import urllib.request
from datetime import datetime
from pathlib import Path

from understudy.dataset import (CONTEXT_TURNS, GAP_MINUTES, VOICE_EXAMPLES, Msg, any_of, clean, display_name,
                                 merge_bursts, render)
from understudy.people import fold, read as read_people
from understudy.prompt import build_messages

log = logging.getLogger("understudy")

HISTORY = 40        # raw messages; bursts and the gap cut bring it down to a conversation
MAX_LINES = 5       # real bursts rarely run longer; loops always do

RE_PLACEHOLDER = re.compile(r"\[(?:link [^\]]*|[a-z]+)\]")
REROLLS = 2         # extra tries for a reply that matches --downweight


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


def own_lines(persona: str, voices: dict[str, list[str]], profile: list[str]) -> list[str]:
    """Real one-liners, then the whole profile: the most of what training put in a context."""
    pool = voices.get(fold(persona), [])
    return [*random.sample(pool, min(VOICE_EXAMPLES, len(pool))), *profile]


def server_reply(url: str, messages: list[dict], prefill: str = "") -> str:
    """Any OpenAI-compatible endpoint: llama-server, Ollama, LM Studio.

    A prefill is the start of the reply, written for the model to continue."""
    if prefill:
        messages = [*messages, {"role": "assistant", "content": prefill}]
    # dry_* is llama-server's repeat-from-context penalty; without it a persona echoes its
    # own previous line verbatim. Other servers ignore the extra fields
    body = json.dumps({"model": "understudy", "messages": messages, "temperature": 0.8,
                       "top_p": 0.95, "max_tokens": 200, "repeat_penalty": 1.1,
                       "dry_multiplier": 0.8, "dry_base": 1.75, "dry_allowed_length": 2}).encode()
    req = urllib.request.Request(f"{url.rstrip('/')}/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return with_prefill(json.load(resp)["choices"][0]["message"]["content"].strip(), prefill)


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
    ap = argparse.ArgumentParser(prog="understudy bot", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voices", type=Path, default=Path("private/data/voices.json"), metavar="FILE")
    ap.add_argument("--llm-url", default=os.environ.get("LLM_URL", "http://localhost:8080"),
                    help="OpenAI-compatible server (default http://localhost:8080)")
    ap.add_argument("--channel", action="append", metavar="NAME",
                    default=[c.strip() for c in os.environ.get("CHANNELS", "").split(",") if c.strip()],
                    help="channel to reply in; repeat for each (default: all)")
    ap.add_argument("--converse-every", type=float, default=0, metavar="MINUTES",
                    help="start a conversation among the personas this often (default 0: never)")
    ap.add_argument("--converse-length", type=int, default=20, metavar="N", help="messages per conversation")
    ap.add_argument("--converse-channel", metavar="NAME", help="channel the conversations happen in")
    ap.add_argument("--downweight", action="append", default=[], metavar="REGEX",
                    help=f"a matching reply is generated again, up to {REROLLS} times, then dropped")
    ap.set_defaults(people={})  # the config's [persona.*] and [aliases] tables
    return ap


def run(args: argparse.Namespace) -> int:

    import discord

    people = read_people(args.people)
    aliases = people.aliases
    tokens = {p.name: p.token for p in people.personas if p.token}
    profiles = {p.name: p.profile for p in people.personas}
    personas = list(tokens)
    if not personas:
        raise SystemExit('no persona has a token: add token = "..." to its [persona.NAME] table')
    voices = json.loads(args.voices.read_text(encoding="utf-8")) if args.voices.exists() else {}
    llm_url = args.llm_url
    channels = set(args.channel)
    avoid = any_of(args.downweight)
    if args.converse_every and not args.converse_channel:
        raise SystemExit("--converse-every needs --converse-channel")
    if args.converse_every and len(personas) < 2:
        raise SystemExit("conversations need at least two personas with a token")

    intents = discord.Intents.default()
    intents.message_content = True
    clients = {p: discord.Client(intents=intents) for p in personas}
    persona_of: dict[int, str] = {}  # bot user id -> persona, filled as each logs in
    locks: dict[int, asyncio.Lock] = {}
    last_person: dict[int, datetime] = {}  # channel id -> when a person last wrote there

    def person_name(user) -> str:
        # the account name, the same everywhere; server nicknames can be aliased to a persona
        return display_name({"name": user.name, "nickname": getattr(user, "global_name", None) or user.name},
                            aliases)

    async def history(channel, since: datetime | None = None) -> list[str]:
        msgs = []
        async for m in channel.history(limit=HISTORY, after=since, oldest_first=False):
            if m.author.bot and m.author.id not in persona_of:
                continue  # other bots, like the training data
            mentions = [{"id": str(u.id), "name": u.name,
                         "nickname": persona_of.get(u.id) or u.display_name} for u in m.mentions]
            text = clean(m.content, mentions, aliases)
            if not text and not m.attachments:
                continue
            author = persona_of.get(m.author.id) or person_name(m.author)
            msgs.append(Msg(id=str(m.id), ts=m.created_at, author=author, content=text,
                            channel=channel.name, reply_to=None, had_attachment=bool(m.attachments)))
        return conversation(list(reversed(msgs)))

    async def speak(channel, persona: str, prefill: str = "", since: datetime | None = None) -> str:
        async with locks.setdefault(channel.id, asyncio.Lock()):
            async with channel.typing():
                turns = await history(channel, since)
                messages = build_messages(persona, turns[-CONTEXT_TURNS:],
                                          own_lines(persona, voices, profiles[persona]))
                speakers = {fold(p) for p in personas} | {fold(t.split(": ", 1)[0]) for t in turns}
                for _ in range(1 + REROLLS):
                    raw = await asyncio.to_thread(server_reply, llm_url, messages, prefill)
                    # matched on the raw reply: cleanup removes some of what the patterns look for
                    if not (avoid and avoid.search(raw)):
                        break
                    log.info("%s's reply matched --downweight, generating again", persona)
                else:
                    return ""
            text = clean_reply(raw, persona, speakers)
            if text:
                await channel.send(text, allowed_mentions=discord.AllowedMentions.none())
                log.info("%s replied in #%s", persona, channel.name)
            return text

    async def converse(channel_id: int) -> None:
        started = discord.utils.utcnow()
        persona = random.choice(personas)
        # opens by turning to someone, so it starts its own thread instead of carrying on the last one
        prefill = f"@{next_speaker('', persona, personas, random)} "
        for _ in range(args.converse_length):
            if last_person.get(channel_id, started) > started:
                log.info("conversation stops: a person wrote")
                return
            # only what was said since it started: each conversation begins blank
            text = await speak(clients[persona].get_channel(channel_id), persona, prefill, started)
            if not text:
                log.info("conversation stops: %s had nothing to say", persona)
                return
            prefill = ""
            persona = next_speaker(text, persona, personas, random)
            await asyncio.sleep(random.uniform(3, 8))  # a beat between people, not a script

    async def converse_every(minutes: float, name: str) -> None:
        first = clients[personas[0]]
        # wait_until_ready() raises before a client has logged in, and the logins are staggered
        while not all(c.is_ready() for c in clients.values()):
            await asyncio.sleep(1)
        channel = discord.utils.get(first.get_all_channels(), name=name, type=discord.ChannelType.text)
        if channel is None:
            print(f"no text channel named {name}; no conversations")
            return
        while True:
            # runs under gather with the bots: an error escaping here would stop all of them
            try:
                await converse(channel.id)
            except Exception:
                log.exception("conversation failed")
            await asyncio.sleep(minutes * 60)

    def wire(persona: str, client) -> None:
        @client.event
        async def on_ready():
            persona_of[client.user.id] = persona
            print(f"{persona} logged in as {client.user}")

        @client.event
        async def on_message(message):
            # only people trigger replies, or the personas would answer each other forever
            if message.author.bot or not isinstance(message.channel, discord.TextChannel):
                return
            last_person[message.channel.id] = message.created_at
            if channels and message.channel.name not in channels:
                return
            # typing @name often autocompletes to the role Discord creates for the bot
            mentioned = any(u.id == client.user.id for u in message.mentions) or any(
                r.tags and r.tags.bot_id == client.user.id for r in message.role_mentions)
            if mentioned or persona in named_personas(message.content, personas):
                # each bot posts through its own connection, so the channel comes from this client
                await speak(client.get_channel(message.channel.id), persona)

    for persona, client in clients.items():
        wire(persona, client)

    async def start(persona: str, client, delay: float) -> None:
        # a small gap so the logins don't all hit Discord in the same instant
        await asyncio.sleep(delay)
        # one bad token or missing intent shouldn't take the other personas down with it
        try:
            await client.start(tokens[persona])
        except Exception as e:
            print(f"{persona} could not connect: {type(e).__name__}: {e}")

    async def start_all():
        # client.run() sets this up itself; start() doesn't, and connection trouble goes silent
        discord.utils.setup_logging()
        tasks = [start(p, c, i) for i, (p, c) in enumerate(clients.items())]
        if args.converse_every:
            tasks.append(converse_every(args.converse_every, args.converse_channel))
        await asyncio.gather(*tasks)

    asyncio.run(start_all())
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
