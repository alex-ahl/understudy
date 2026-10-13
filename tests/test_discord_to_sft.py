import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "discord_to_sft.py"
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


def run(messages, *flags):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "export.json").write_text(json.dumps(
            {"channel": {"name": "general"}, "messages": messages}))
        proc = subprocess.run(
            [sys.executable, SCRIPT, tmp / "export.json", "-o", tmp / "out", *flags],
            capture_output=True, text=True)
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
        self.assertEqual(ctx, "carol: first thing said here\ndave: second thing said here")

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

    def test_aliases_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
            fh.write("# real=fake\ndave = pat\n\nDavid=pat  # full name too\n")
        _, rows = run([
            msg(1, "carol", "anyone seen david lately", 0 * M),
            msg(2, "dave", "im right here obviously", 1 * M),
            msg(3, "bob", "dave and david are the same guy", 2 * M),
        ], "--persona", "bob", "--aliases", fh.name)
        Path(fh.name).unlink()
        [(target, ctx)] = by_target(rows).items()
        self.assertEqual(target, "pat and pat are the same guy")
        self.assertEqual(ctx, "carol: anyone seen pat lately\npat: im right here obviously")

    def test_alias_ignores_accents_and_stray_marks(self):
        _, rows = run([
            msg(1, "carol", "has anyone seen rené", 0 * M),
            msg(2, "Rêne", "re´ne is here, RENE is fine", 1 * M),
            msg(3, "bob", "rene, renes and lorene", 2 * M),
        ], "--persona", "bob", "--alias", "rene=pat")
        [(target, ctx)] = by_target(rows).items()
        self.assertEqual(target, "pat, pat and pat")
        self.assertEqual(ctx, "carol: has anyone seen pat\npat: pat is here, pat is fine")

    def test_aliases_file_marks_personas(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
            fh.write("*robert = bob\n*carol\ndave=pat\n")
        _, rows = run([
            msg(1, "robert", "robert opens the chat", 0 * M),
            msg(2, "dave", "dave chimes in here", 1 * M),
            msg(3, "carol", "carol answers them both", 2 * M),
            msg(4, "robert", "robert closes the chat", 3 * M),
        ], "--aliases", fh.name)
        Path(fh.name).unlink()
        systems = sorted(r["messages"][0]["content"] for r in rows)
        self.assertEqual(systems, ["Reply as `bob` in #general.", "Reply as `carol` in #general."])

    def test_long_alias_replaces_any_word_containing_it(self):
        _, rows = run([
            msg(1, "carol", "where is everyone", 0 * M),
            msg(2, "dave", "no clue at all", 1 * M),
            msg(3, "bob", "roberts car, bigrobert and per", 2 * M),
        ], "--persona", "bob", "--alias", "robert=pat", "--alias", "per=kim")
        self.assertIn("pat car, pat and kim", by_target(rows))

    def test_longer_alias_wins_regardless_of_order(self):
        _, rows = run([
            msg(1, "carol", "where is everyone", 0 * M),
            msg(2, "dave", "no clue at all", 1 * M),
            msg(3, "bob", "robert and roberta both came", 2 * M),
        ], "--persona", "bob", "--alias", "robert=pat", "--alias", "roberta=kim")
        self.assertIn("pat and kim both came", by_target(rows))

    def test_quoted_alias_matches_whole_words_only(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
            fh.write('"acme"=[workplace]\n')
        _, rows = run([
            msg(1, "carol", "where is everyone", 0 * M),
            msg(2, "dave", "no clue at all", 1 * M),
            msg(3, "bob", "Acme and Acmes office, not acmeville", 2 * M),
        ], "--persona", "bob", "--aliases", fh.name)
        Path(fh.name).unlink()
        self.assertIn("[workplace] and [workplace] office, not acmeville", by_target(rows))

    def test_scrubs_contact_details(self):
        _, rows = run([
            msg(1, "carol", "how do i reach you", 0 * M),
            msg(2, "dave", "yeah what's your info", 1 * M),
            msg(3, "bob", "mail bob@example.com or join discord.gg/abc123", 2 * M),
        ], "--persona", "bob")
        [target] = by_target(rows)
        self.assertEqual(target, "mail [email] or join [invite]")

    def test_scrubs_street_addresses(self):
        _, rows = run([
            msg(1, "carol", "where do you live again", 0 * M),
            msg(2, "dave", "yeah send the address", 1 * M),
            msg(3, "bob", "Exempelvägen 12B, 123 45 Småby. or storgatan 3", 2 * M),
        ], "--persona", "bob", "--alias", "exempelvägen=[place]")
        self.assertIn("[address]. or [address]", by_target(rows))

    def test_address_pattern_leaves_plain_words_alone(self):
        _, rows = run([
            msg(1, "carol", "how was the trip", 0 * M),
            msg(2, "dave", "tell us everything", 1 * M),
            msg(3, "bob", "vägen var lång, 12 mil på gatan", 2 * M),
        ], "--persona", "bob")
        self.assertIn("vägen var lång, 12 mil på gatan", by_target(rows))

    def test_links_keep_only_their_domain(self):
        _, rows = run([
            msg(1, "carol", "got any pics from the trip", 0 * M),
            msg(2, "dave", "yes send the album please", 1 * M),
            msg(3, "bob", "here (https://photos.example.com/abc123?x=1). and www.shop.se/item/9! "
                          "spotify:user:someone:playlist:1 <a href=\"http://a.example/x\">", 2 * M),
        ], "--persona", "bob")
        self.assertIn("here ([link photos.example.com]). and [link shop.se]! "
                      "[link spotify] <a href=\"[link a.example]\">", by_target(rows))

    def test_scrubs_meeting_passwords(self):
        _, rows = run([
            msg(1, "carol", "what is the meeting info", 0 * M),
            msg(2, "dave", "check the invite below", 1 * M),
            msg(3, "bob", "Password: 1CzQ9y and lösenord=hunter2 for the call", 2 * M),
        ], "--persona", "bob")
        self.assertIn("Password: [password] and lösenord=[password] for the call", by_target(rows))

    def test_profiles_go_into_the_system_prompt(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
            fh.write("[bob]\nAlways hungry.\n")
        _, rows = run([
            msg(1, "carol", "where is everyone", 0 * M),
            msg(2, "dave", "no clue at all", 1 * M),
            msg(3, "bob", "probably getting food again", 2 * M),
        ], "--persona", "bob", "--profiles", fh.name)
        Path(fh.name).unlink()
        [row] = rows
        self.assertEqual(row["messages"][0]["content"],
                         "Reply as `bob` in #general.\nAbout bob: Always hungry.")

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
                [sys.executable, SCRIPT, tmp, "--persona", "bob", "-o", tmp / "out"],
                capture_output=True, text=True)
        self.assertRegex(proc.stdout, r"\n\s+2  bob\n")


if __name__ == "__main__":
    unittest.main()
