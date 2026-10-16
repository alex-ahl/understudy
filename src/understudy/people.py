"""Who is who: the [persona.NAME] and [aliases] tables of understudy.toml.

    [persona.bob]
    names = ["robert_1987", "Robban"]   # how they appear in Discord; all become "bob"
    profile = ["gaming again tonight"]  # lines in their own voice, most important last
    token = "..."                       # their bot's token

    [aliases]                           # everyone else who gets renamed
    carl_xx = "carl"

Names match whole words, plus a possessive s ("roberts" becomes "bob"), with
case, accents and stray marks like ´ ignored.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass
class Persona:
    name: str
    names: list[str] = field(default_factory=list)
    profile: list[str] = field(default_factory=list)
    token: str | None = None


@dataclass
class People:
    personas: list[Persona]
    aliases: dict[str, str]     # folded real name -> pseudonym, personas' names included


def fold_char(ch: str) -> str:
    """Lowercase, accents stripped; stray accent marks like ´ fold to nothing."""
    if unicodedata.category(ch) == "Sk":
        return ""
    base = unicodedata.normalize("NFD", ch)
    return "".join(c for c in base if not unicodedata.combining(c)).lower()


def fold(text: str) -> str:
    return "".join(fold_char(ch) for ch in text)


def read(tables: dict, personas: Sequence[str] = (), aliases: Sequence[str] = ()) -> People:
    """People from the config's tables, plus --persona NAME and --alias REAL=FAKE.

    Raises ValueError on a malformed --alias, or a name renamed to two different pseudonyms."""
    found = [Persona(name, t.get("names", []), t.get("profile", []), t.get("token"))
             for name, t in tables.get("persona", {}).items()]
    found += [Persona(p) for p in personas if fold(p) not in {fold(f.name) for f in found}]

    mapping: dict[str, str] = {}

    def add(real: str, fake: str) -> None:
        key = fold(real.strip())
        if mapping.get(key, fake) != fake:
            raise ValueError(f"{real!r} is renamed to both {mapping[key]!r} and {fake!r}")
        mapping[key] = fake

    for p in found:
        for name in p.names:
            add(name, p.name)
    for real, fake in tables.get("aliases", {}).items():
        add(real, fake)
    for pair in aliases:
        real, _, fake = pair.partition("=")
        if not real.strip() or not fake.strip():
            raise ValueError(f"alias wants REAL=FAKE, got {pair!r}")
        add(real, fake.strip())
    return People(found, mapping)


def replace_folded(content: str, real: str, fake: str) -> str:
    """Replace `real` with `fake` as a whole word, matching on folded text."""
    # not inside longer words: a name in another word is usually just that word ("per" in "super")
    folded, origin = [], []
    for i, ch in enumerate(content):
        for f in fold_char(ch):
            folded.append(f)
            origin.append(i)
    folded = "".join(folded)

    spans = [(origin[m.start()], origin[m.end() - 1] + 1)
             for m in re.finditer(rf"(?<!\w){re.escape(real)}s?(?!\w)", folded)]
    for start, end in reversed(spans):
        content = content[:start] + fake + content[end:]
    return content


def rename(content: str, aliases: dict[str, str]) -> str:
    # longest first, so "roberta" is replaced before "robert" can claim it
    for real in sorted(aliases, key=len, reverse=True):
        content = replace_folded(content, real, aliases[real])
    return content
