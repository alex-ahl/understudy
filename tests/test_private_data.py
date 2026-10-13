"""Fails when a tracked file contains a name, profile line or token from the local,
gitignored private files. Skipped where those files don't exist, as in a fresh clone."""

import re
import subprocess
import unittest
from pathlib import Path

from understudy.dataset import fold

ROOT = Path(__file__).resolve().parent.parent
MIN_NAME = 3        # shorter names match too many ordinary words
MIN_LINE = 15


def entries(name: str) -> list[str]:
    path = ROOT / name
    if not path.exists():
        return []
    return [line.split("#", 1)[0].strip() for line in path.read_text(encoding="utf-8").splitlines()]


def secrets() -> tuple[set[str], set[str]]:
    """Names to match as whole words, and lines and tokens to match anywhere, all folded."""
    names, verbatim = set(), set()
    for line in entries("aliases.txt"):
        for part in re.split(r"[=,]", line.lstrip("*")):
            part = part.strip().strip('"')
            if len(part) >= MIN_NAME and not part.startswith("["):
                names.add(fold(part))
    for line in entries("profiles.txt"):
        if line.startswith("[") and line.endswith("]"):
            names.add(fold(line[1:-1].strip()))
        elif len(line) >= MIN_LINE:
            verbatim.add(fold(line))
    for line in entries("bots.txt"):
        persona, _, token = line.partition("=")
        names.add(fold(persona.strip()))
        if token.strip():
            verbatim.add(fold(token.strip()))
    return {n for n in names if len(n) >= MIN_NAME}, verbatim


def tracked() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True)
    return [ROOT / p for p in out.stdout.split("\0") if p] if out.returncode == 0 else []


class PrivateData(unittest.TestCase):
    def test_tracked_files_hold_no_private_data(self):
        names, verbatim = secrets()
        files = tracked()
        if not (names or verbatim) or not files:
            self.skipTest("no private files or no git checkout")
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
