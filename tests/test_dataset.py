import json
import random
import re
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from understudy.dataset import (VOICE_EXAMPLES, Example, any_of, cap, downweight, has_link, own_lines,
                                voice_pool)

# run from a temp dir: the config is read from private/ under the working directory.
# --no-scrub: scrubbing loads its models per run; test_scrub covers it
DATASET = [sys.executable, "-m", "understudy", "dataset"]
T0 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def msg(i, author, content, at, **extra):
    return {
        "id": str(i),
        "type": "Default",
        "timestamp": (T0 + at).isoformat(),
        "author": {"id": author, "name": author, "isBot": False},
        "content": content,
        "attachments": [],
        "mentions": [],
        **extra,
    }


def run(messages, *flags, config=None):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        if config:
            (tmp / "private").mkdir()
            (tmp / "private" / "understudy.toml").write_text(config)
        (tmp / "export.json").write_text(json.dumps(
            {"channel": {"name": "general"}, "messages": messages}))
        proc = subprocess.run(
            [*DATASET, tmp / "export.json", "-o", tmp / "out", "--no-scrub", *flags],
            capture_output=True, text=True, cwd=tmp)
        rows = []
        for name in ("train", "val"):
            path = tmp / "out" / f"{name}.jsonl"
            if path.exists():
                rows += [json.loads(line) for line in path.read_text().splitlines()]
        return proc, rows


def by_target(rows):
    return {r["messages"][2]["content"]: r["messages"][1]["content"] for r in rows}


S = timedelta(seconds=1)
M = timedelta(minutes=1)


class DiscordToSft(unittest.TestCase):
    def test_context_is_chronological_and_never_includes_later_messages(self):
        proc, rows = run([
            msg(1, "carol", "first thing said here", 0 * M),
            msg(2, "dave", "second thing said here", 1 * M),
            msg(3, "bob", "bob replies to all that", 2 * M),
            msg(4, "carol", "a later message nobody saw", 3 * M),
        ], "--persona", "bob")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        ctx = by_target(rows)["bob replies to all that"]
        self.assertEqual(ctx, "carol: first thing said here\ndave: second thing said here\nbob:")

    def test_context_stops_at_a_conversation_gap(self):
        _, rows = run([
            msg(1, "carol", "old conversation entirely", 0 * M),
            msg(2, "carol", "new conversation starts", 30 * M),
            msg(3, "dave", "and continues onward", 31 * M),
            msg(4, "bob", "bob joins the new one", 32 * M),
        ], "--persona", "bob")
        self.assertNotIn("old conversation", by_target(rows)["bob joins the new one"])

    def test_bursts_from_one_author_merge_into_one_turn(self):
        _, rows = run([
            msg(1, "carol", "hello there everyone", 0 * M),
            msg(2, "dave", "hi carol how are you", 1 * M),
            msg(3, "bob", "wait", 2 * M),
            msg(4, "bob", "actually hold on a second", 2 * M + 20 * S),
        ], "--persona", "bob")
        self.assertIn("wait\nactually hold on a second", by_target(rows))

    def test_alias_rewrites_authors_and_plain_text_mentions(self):
        _, rows = run([
            msg(1, "carol", "has anyone seen dave today", 0 * M),
            msg(2, "dave", "im right here obviously", 1 * M),
            msg(3, "bob", "dave is always lurking", 2 * M),
        ], "--persona", "bob", "--alias", "dave=pat")
        [(target, ctx)] = by_target(rows).items()
        self.assertEqual(target, "pat is always lurking")
        self.assertIn("pat: im right here", ctx)
        self.assertNotIn("dave", ctx)

    def test_alias_rewrites_rendered_mentions(self):
        _, rows = run([
            msg(1, "carol", "where did everyone go", 0 * M),
            msg(2, "dave", "no idea honestly", 1 * M),
            msg(3, "bob", "@dave you said you'd be here", 2 * M),
        ], "--persona", "bob", "--alias", "dave=pat")
        self.assertIn("@pat you said you'd be here", by_target(rows))

    def test_aliases_table(self):
        _, rows = run([
            msg(1, "carol", "anyone seen david lately", 0 * M),
            msg(2, "dave", "im right here obviously", 1 * M),
            msg(3, "bob", "dave and david are the same guy", 2 * M),
        ], "--persona", "bob", config='[aliases]\ndave = "pat"\nDavid = "pat"  # full name too\n')
        [(target, ctx)] = by_target(rows).items()
        self.assertEqual(target, "pat and pat are the same guy")
        self.assertEqual(ctx, "carol: anyone seen pat lately\npat: im right here obviously\nbob:")

    def test_alias_ignores_accents_and_stray_marks(self):
        _, rows = run([
            msg(1, "carol", "has anyone seen chloé", 0 * M),
            msg(2, "Chloë", "chlo´e is here, CHLOE is fine", 1 * M),
            msg(3, "bob", "chloe, and chloes bike too", 2 * M),
        ], "--persona", "bob", "--alias", "chloe=pat")
        [(target, ctx)] = by_target(rows).items()
        self.assertEqual(target, "pat, and pat bike too")
        self.assertEqual(ctx, "carol: has anyone seen pat\npat: pat is here, pat is fine\nbob:")

    def test_persona_tables_are_the_personas(self):
        _, rows = run([
            msg(1, "robert", "robert opens the chat", 0 * M),
            msg(2, "dave", "dave chimes in here", 1 * M),
            msg(3, "carol", "carol answers them both", 2 * M),
            msg(4, "robert", "robert closes the chat", 3 * M),
        ], config='[persona.bob]\nnames = ["robert"]\n[persona.carol]\n[aliases]\ndave = "pat"\n')
        systems = sorted(r["messages"][0]["content"] for r in rows)
        self.assertEqual(systems, ["Reply as `bob`.", "Reply as `carol`."])

    def test_alias_keeps_the_possessive_but_not_longer_words(self):
        _, rows = run([
            msg(1, "carol", "where is everyone", 0 * M),
            msg(2, "dave", "no clue at all", 1 * M),
            msg(3, "bob", "roberts car, bigrobert and per", 2 * M),
        ], "--persona", "bob", "--alias", "robert=pat", "--alias", "per=kim")
        self.assertIn("pat car, bigrobert and kim", by_target(rows))

    def test_longer_alias_wins_regardless_of_order(self):
        _, rows = run([
            msg(1, "carol", "where is everyone", 0 * M),
            msg(2, "dave", "no clue at all", 1 * M),
            msg(3, "bob", "robert and roberta both came", 2 * M),
        ], "--persona", "bob", "--alias", "robert=pat", "--alias", "roberta=kim")
        self.assertIn("pat and kim both came", by_target(rows))

    def test_alias_matches_whole_words_only(self):
        _, rows = run([
            msg(1, "carol", "where is everyone", 0 * M),
            msg(2, "dave", "no clue at all", 1 * M),
            msg(3, "bob", "Acme and Acmes office, not acmeville", 2 * M),
        ], "--persona", "bob", "--alias", "acme=[workplace]")
        self.assertIn("[workplace] and [workplace] office, not acmeville", by_target(rows))

    def test_profiles_go_into_the_context_not_the_system_prompt(self):
        _, rows = run([
            msg(1, "carol", "where is everyone", 0 * M),
            msg(2, "dave", "no clue at all", 1 * M),
            msg(3, "bob", "probably getting food again", 2 * M),
        ], config='[persona.bob]\nprofile = ["Always hungry."]\n')
        [row] = rows
        self.assertEqual(row["messages"][0]["content"], "Reply as `bob`.")
        own = [l for l in row["messages"][1]["content"].splitlines() if l.startswith("bob: ")]
        self.assertLessEqual(set(own), {"bob: Always hungry."})

    def test_persona_counts_sum_across_channels(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for channel in ("general", "random"):
                (tmp / f"{channel}.json").write_text(json.dumps({
                    "channel": {"name": channel},
                    "messages": [
                        msg(f"{channel}1", "carol", "something to reply to", 0 * M),
                        msg(f"{channel}2", "dave", "something else as well", 1 * M),
                        msg(f"{channel}3", "bob", "bob has thoughts on it", 2 * M),
                    ],
                }))
            proc = subprocess.run(
                [*DATASET, tmp, "--persona", "bob", "-o", tmp / "out", "--no-scrub"],
                capture_output=True, text=True, cwd=tmp)
        self.assertRegex(proc.stdout, r"\n\s+2  bob\n")

    def test_link_replies_are_dropped(self):
        _, rows = run([
            msg(1, "carol", "where is everyone", 0 * M),
            msg(2, "dave", "no clue at all", 1 * M),
            msg(3, "bob", "look at https://example.com first", 2 * M),
        ], "--persona", "bob")
        self.assertEqual(rows, [])

    def test_voices_file_holds_each_personas_one_liners(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "export.json").write_text(json.dumps({"channel": {"name": "general"}, "messages": [
                msg(1, "carol", "where is everyone", 0 * M),
                msg(2, "dave", "no clue at all", 1 * M),
                msg(3, "bob", "probably getting food again", 2 * M),
            ]}))
            subprocess.run([*DATASET, tmp / "export.json", "-o", tmp / "out", "--no-scrub",
                            "--persona", "bob"], capture_output=True, cwd=tmp)
            voices = json.loads((tmp / "out" / "voices.json").read_text())
        # the only example may land in val, which never feeds the pool
        self.assertIn(voices, ({"bob": ["probably getting food again"]}, {}))


class Always(random.Random):
    """random() pinned, so keep-or-drop decisions are known."""

    def __init__(self, value):
        super().__init__(0)
        self.value = value

    def random(self):
        return self.value


def example(reply, *lines, persona="bob"):
    return Example(persona, list(lines), reply)


class Shaping(unittest.TestCase):
    def test_links(self):
        for text in ("see [link youtube.com]", "https://x.io/a", "www.x.io"):
            self.assertTrue(has_link(text), text)
        self.assertFalse(has_link("no links [place] here"))

    def test_downweight_patterns_match_any_ignoring_case(self):
        pattern = any_of(["tonight", r"\bmeet up\b"])
        self.assertTrue(pattern.search("Shall we MEET UP?"))
        self.assertTrue(pattern.search("tonight then"))
        self.assertFalse(pattern.search("plans"))
        self.assertIsNone(any_of([]))

    def test_downweight_drops_matches_but_never_the_line_replied_to(self):
        pattern = re.compile("tonight")
        examples = [example("sure, tonight works", "carol: hi", "dave: tonight?"),
                    example("not me", "carol: tonight?", "dave: and you?", "carol: tonight?")]
        kept = downweight(examples, pattern, Always(0.99))
        self.assertEqual(kept, [example("not me", "dave: and you?", "carol: tonight?")])
        self.assertEqual(downweight(examples, pattern, Always(0.0)), examples)

    def test_cap_samples_each_persona_down(self):
        examples = [example(f"bob {i}") for i in range(5)] + [example("alice", persona="alice")]
        capped = cap(examples, 2, random.Random(0))
        self.assertEqual(sum(e.persona == "bob" for e in capped), 2)
        self.assertIn(example("alice", persona="alice"), capped)

    def test_voice_pool_takes_plain_one_liners(self):
        examples = [example("a plain one-liner here"), example("two\nlines here"), example("short"),
                    example("see [link x.io] now"), example("we meet tonight then"),
                    example("by alice and nobody else", persona="Alice")]
        self.assertEqual(voice_pool(examples, re.compile("tonight")),
                         {"bob": ["a plain one-liner here"], "alice": ["by alice and nobody else"]})

    def test_own_lines_never_repeat_the_reply_and_keep_profile_order(self):
        pool = ["the reply itself", "other one", "another one", "third one", "fourth one"]
        profile = ["first profile line", "second profile line", "last profile line"]
        seen_empty = False
        for seed in range(50):
            lines = own_lines("the reply itself", pool, profile, random.Random(seed))
            voices = [l for l in lines if l not in profile]
            self.assertNotIn("the reply itself", lines)
            self.assertLessEqual(len(voices), VOICE_EXAMPLES)
            self.assertEqual(lines[len(voices):], [l for l in profile if l in lines])
            seen_empty |= not lines
        self.assertTrue(seen_empty)


if __name__ == "__main__":
    unittest.main()
