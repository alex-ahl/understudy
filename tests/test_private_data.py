"""Fails when a tracked file contains a name, profile line or token from the local,
gitignored private/understudy.toml. Skipped where it doesn't exist, as in a fresh clone."""

import re
import subprocess
import tomllib
import unittest
from pathlib import Path

from understudy.people import fold

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "private" / "understudy.toml"
MIN_NAME = 3        # shorter names match too many ordinary words
MIN_LINE = 15


def secrets() -> tuple[set[str], set[str]]:
    """Names to match as whole words, and lines and tokens to match anywhere, all folded."""
    if not CONFIG.exists():
        return set(), set()
    config = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
    names, verbatim = set(), set()
    for persona, table in config.get("persona", {}).items():
        names |= {persona, *table.get("names", [])}
        verbatim |= {line for line in table.get("profile", []) if len(line) >= MIN_LINE}
        if table.get("token"):
            verbatim.add(table["token"])
    for real, fake in config.get("aliases", {}).items():
        names |= {real, fake}
    names = {fold(n) for n in names if not n.startswith("[")}
    return {n for n in names if len(n) >= MIN_NAME}, {fold(v) for v in verbatim}


def tracked() -> list[Path]:
    # safe.directory: in a container the mounted repo belongs to another user, and git refuses it
    out = subprocess.run(["git", "-c", "safe.directory=*", "ls-files", "-z"], cwd=ROOT,
                         capture_output=True, text=True)
    return [ROOT / p for p in out.stdout.split("\0") if p] if out.returncode == 0 else []


class PrivateData(unittest.TestCase):
    def test_tracked_files_hold_no_private_data(self):
        names, verbatim = secrets()
        files = tracked()
        if not (names or verbatim) or not files:
            self.skipTest("no private/understudy.toml or no git checkout")
        word = re.compile(r"(?<!\w)(" + "|".join(map(re.escape, sorted(names, key=len, reverse=True)))
                          + r")(?!\w)") if names else None
        hits = []
        for path in files:
            try:
                text = fold(path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, FileNotFoundError):
                continue
            rel = path.relative_to(ROOT)
            for n, line in enumerate(text.splitlines(), 1):
                found = {m.group(1) for m in word.finditer(line)} if word else set()
                found |= {v for v in verbatim if v in line}
                hits += [f"{rel}:{n}: {f}" for f in sorted(found)]
        self.assertEqual(hits, [], "private data in tracked files:\n" + "\n".join(hits))


if __name__ == "__main__":
    unittest.main()
