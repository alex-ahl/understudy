import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from understudy.bot import (build_messages, clean_reply, conversation, load_personas, named_personas,
                            read_tokens)
from understudy.dataset import Msg

T0 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def msg(author, content, minutes):
    return Msg(id=str(minutes), ts=T0 + timedelta(minutes=minutes), author=author,
               content=content, channel="general", reply_to=None, had_attachment=False)


def write(text: str) -> Path:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write(text)
    return Path(fh.name)


class Bot(unittest.TestCase):
    def test_personas_are_their_pseudonyms(self):
        path = write('*robert=bob\n*alice\n"carl"=kim\n')
        aliases, personas = load_personas(path)
        path.unlink()
        self.assertEqual(personas, ["bob", "alice"])
        self.assertEqual(aliases["robert"], "bob")

    def test_reads_one_token_per_persona(self):
        path = write("# tokens\nBob=abc.def\n\nalice = xyz  # spare\n")
        tokens = read_tokens(path)
        path.unlink()
        self.assertEqual(tokens, {"bob": "abc.def", "alice": "xyz"})

    def test_named_personas_match_whole_words_in_order(self):
        personas = ["bob", "alice"]
        self.assertEqual(named_personas("Alice, and you bob?", personas), ["alice", "bob"])
        self.assertEqual(named_personas("bobby isn't coming", personas), [])

    def test_conversation_merges_bursts_and_stops_at_a_gap(self):
        turns = conversation([
            msg("carol", "old topic", 0),
            msg("bob", "new topic starts", 30),
            msg("bob", "still bob", 30),
            msg("carol", "answering bob", 31),
        ])
        self.assertEqual(turns, ["bob: new topic starts\nstill bob", "carol: answering bob"])

    def test_profile_goes_in_near_the_end_as_the_personas_lines(self):
        turns = [f"p{i}: line {i}" for i in range(6)]
        system, user = build_messages("bob", "general", turns, {"bob": "Likes trains.\nHates Mondays."})
        self.assertEqual(user["content"].splitlines(),
                         ["p0: line 0", "p1: line 1", "p2: line 2",
                          "bob: Likes trains.", "bob: Hates Mondays.",
                          "p3: line 3", "p4: line 4", "p5: line 5"])
        self.assertIn("About bob: Likes trains. Hates Mondays.", system["content"])

    def test_reply_keeps_only_the_personas_own_clean_lines(self):
        raw = "bob: sure thing\ncheck [link example.com] out\nsure thing\nalice: no way"
        self.assertEqual(clean_reply(raw, "bob", {"bob", "alice"}), "sure thing\ncheck out")

    def test_reply_is_scrubbed(self):
        out = clean_reply("call me on 070-123 45 67, or mail a@b.se", "bob", {"bob"})
        self.assertNotIn("070-123", out)
        self.assertNotIn("a@b.se", out)


if __name__ == "__main__":
    unittest.main()
