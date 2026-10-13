import unittest
from datetime import datetime, timedelta, timezone

from understudy.bot import clean_reply, conversation, named_personas, own_lines
from understudy.dataset import Msg

T0 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def msg(author, content, minutes):
    return Msg(id=str(minutes), ts=T0 + timedelta(minutes=minutes), author=author,
               content=content, channel="general", reply_to=None, had_attachment=False)


class Bot(unittest.TestCase):
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

    def test_own_lines_are_voice_lines_then_the_whole_profile(self):
        voices = {"bob": ["one", "two", "three", "four"]}
        lines = own_lines("Bob", voices, ["Likes trains.", "Hates Mondays."])
        self.assertEqual(lines[3:], ["Likes trains.", "Hates Mondays."])
        self.assertLessEqual(set(lines[:3]), set(voices["bob"]))
        self.assertEqual(own_lines("carol", voices, []), [])

    def test_reply_keeps_only_the_personas_own_clean_lines(self):
        raw = "bob: sure thing\ncheck [link example.com] out\nsure thing\nalice: no way"
        self.assertEqual(clean_reply(raw, "bob", {"bob", "alice"}), "sure thing\ncheck out")


if __name__ == "__main__":
    unittest.main()
