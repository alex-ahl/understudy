"""Discord bots that let the personas talk in the server, through llama-server.

    understudy bot

Each persona is its own bot account; bots.txt holds one `persona=token` line
per persona. aliases.txt, bots.txt and profiles.txt are read from the working
directory unless given as flags. A persona replies to the channel's recent
conversation when someone @-mentions it or writes its name in a message.

--llm-url and --channel default to the LLM_URL and CHANNELS (comma-separated)
environment variables.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import urllib.request
from pathlib import Path

from understudy.dataset import (CONTEXT_TURNS, GAP_MINUTES, Msg, build_aliases, clean, display_name,
                                 fold, merge_bursts, read_aliases_file, render, unquote)
from understudy.profiles import read_profiles, system_prompt

log = logging.getLogger("understudy")

HISTORY = 40        # raw messages; bursts and the gap cut bring it down to a conversation
PROFILE_TAIL = 3    # profile lines go this many turns before the end of the context
MAX_LINES = 5       # real bursts rarely run longer; loops always do

RE_PLACEHOLDER = re.compile(r"\[(?:link [^\]]*|[a-z]+)\]")


def load_personas(aliases_path: Path) -> tuple[dict[str, str], list[str]]:
    """Aliases as the dataset built them, and the personas by their pseudonyms."""
    pairs, marked = read_aliases_file(aliases_path)
    aliases = build_aliases(pairs)
    personas = []
    for name in marked:
        real = fold(unquote(name.strip())[0])
        personas.append(aliases.get(real) or aliases.get(f'"{real}"') or name.strip())
    return aliases, personas


def read_tokens(path: Path) -> dict[str, str]:
    """persona=token lines, # comments; persona names compared folded."""
    tokens = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            persona, _, token = line.partition("=")
            tokens[fold(persona.strip())] = token.strip()
    return tokens


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


def build_messages(persona: str, channel: str, turns: list[str], profiles: dict[str, str]) -> list[dict]:
    """The prompt for one reply, with the profile as the persona's own recent lines."""
    lines = turns[-CONTEXT_TURNS:]
    profile = profiles.get(persona.lower())
    # the adapter reads the system prompt as a name tag but follows the conversation, and near
    # the end of it: profile lines go in as the persona's own messages, a few turns from the end
    if profile:
        cut = max(len(lines) - PROFILE_TAIL, 0)
        lines = [*lines[:cut], *(f"{persona}: {line}" for line in profile.splitlines()), *lines[cut:]]
    return [
        {"role": "system", "content": system_prompt(persona, channel, profiles)},
        {"role": "user", "content": "\n".join(lines)},
    ]


def server_reply(url: str, messages: list[dict]) -> str:
    """Any OpenAI-compatible endpoint: llama-server, Ollama, LM Studio."""
    # dry_* is llama-server's repeat-from-context penalty; without it a persona echoes its
    # own previous line verbatim. Other servers ignore the extra fields
    body = json.dumps({"model": "understudy", "messages": messages, "temperature": 0.8,
                       "top_p": 0.95, "max_tokens": 200, "repeat_penalty": 1.1,
                       "dry_multiplier": 0.8, "dry_base": 1.75, "dry_allowed_length": 2}).encode()
    req = urllib.request.Request(f"{url.rstrip('/')}/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)["choices"][0]["message"]["content"].strip()


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
    ap.add_argument("--aliases", type=Path, default=Path("aliases.txt"), metavar="FILE")
    ap.add_argument("--tokens", type=Path, default=Path("bots.txt"), metavar="FILE")
    ap.add_argument("--profiles", type=Path, default=Path("profiles.txt"), metavar="FILE")
    ap.add_argument("--llm-url", default=os.environ.get("LLM_URL", "http://localhost:8080"),
                    help="OpenAI-compatible server (default http://localhost:8080)")
    ap.add_argument("--channel", action="append", metavar="NAME",
                    default=[c.strip() for c in os.environ.get("CHANNELS", "").split(",") if c.strip()],
                    help="channel to reply in; repeat for each (default: all)")
    return ap


def run(args: argparse.Namespace) -> int:

    import discord

    aliases, all_personas = load_personas(args.aliases)
    tokens = read_tokens(args.tokens)
    personas = [p for p in all_personas if fold(p) in tokens]
    if not personas:
        raise SystemExit(f"{args.tokens} has no token for any persona in {args.aliases}")
    profiles_path = args.profiles
    llm_url = args.llm_url
    channels = set(args.channel)

    intents = discord.Intents.default()
    intents.message_content = True
    clients = {p: discord.Client(intents=intents) for p in personas}
    persona_of: dict[int, str] = {}  # bot user id -> persona, filled as each logs in
    locks: dict[int, asyncio.Lock] = {}

    def person_name(user) -> str:
        # the account name, the same everywhere; server nicknames can be aliased to a persona
        return display_name({"name": user.name, "nickname": getattr(user, "global_name", None) or user.name},
                            aliases)

    async def history(channel) -> list[str]:
        msgs = []
        async for m in channel.history(limit=HISTORY):
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

    async def speak(channel, persona: str) -> None:
        async with locks.setdefault(channel.id, asyncio.Lock()):
            async with channel.typing():
                turns = await history(channel)
                messages = build_messages(persona, channel.name, turns, read_profiles(profiles_path))
                raw = await asyncio.to_thread(server_reply, llm_url, messages)
            speakers = {fold(p) for p in personas} | {fold(t.split(": ", 1)[0]) for t in turns}
            text = clean_reply(raw, persona, speakers)
            if text:
                await channel.send(text, allowed_mentions=discord.AllowedMentions.none())
                log.info("%s replied in #%s", persona, channel.name)

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
            await client.start(tokens[fold(persona)])
        except Exception as e:
            print(f"{persona} could not connect: {type(e).__name__}: {e}")

    async def start_all():
        # client.run() sets this up itself; start() doesn't, and connection trouble goes silent
        discord.utils.setup_logging()
        await asyncio.gather(*(start(p, c, i) for i, (p, c) in enumerate(clients.items())))

    asyncio.run(start_all())
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
