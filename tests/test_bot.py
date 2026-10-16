import argparse
import random
import unittest
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

from understudy.bot import (
    Bots,
    clean_reply,
    conversation,
    named_personas,
    next_speaker,
    own_lines,
    with_prefill,
)
from understudy.chat import Msg
from understudy.people import People, Persona

T0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def msg(author, content, minutes):
    return Msg(
        id=str(minutes),
        ts=T0 + timedelta(minutes=minutes),
        author=author,
        content=content,
        channel="general",
        reply_to=None,
        had_attachment=False,
    )


class Bot(unittest.TestCase):
    def test_named_personas_match_whole_words_in_order(self):
        personas = ["bob", "alice"]
        self.assertEqual(named_personas("Alice, and you bob?", personas), ["alice", "bob"])
        self.assertEqual(named_personas("bobby isn't coming", personas), [])

    def test_conversation_merges_bursts_and_stops_at_a_gap(self):
        turns = conversation(
            [
                msg("carol", "old topic", 0),
                msg("bob", "new topic starts", 30),
                msg("bob", "still bob", 30),
                msg("carol", "answering bob", 31),
            ]
        )
        self.assertEqual(turns, ["bob: new topic starts\nstill bob", "carol: answering bob"])

    def test_own_lines_are_voice_lines_then_the_whole_profile(self):
        voices = {"bob": ["one", "two", "three", "four"]}
        lines = own_lines("Bob", voices, ["Likes trains.", "Hates Mondays."], random.Random(0))
        self.assertEqual(lines[3:], ["Likes trains.", "Hates Mondays."])
        self.assertLessEqual(set(lines[:3]), set(voices["bob"]))
        self.assertEqual(own_lines("carol", voices, [], random.Random(0)), [])

    def test_next_speaker_is_whoever_was_named_else_someone_else(self):
        personas = ["bob", "alice", "carol"]
        self.assertEqual(next_speaker("what do you say, carol?", "bob", personas, random.Random(0)), "carol")
        self.assertIn(next_speaker("bob here again", "bob", personas, random.Random(0)), ("alice", "carol"))
        picks = {next_speaker("nobody named", "bob", personas, random.Random(seed)) for seed in range(20)}
        self.assertEqual(picks, {"alice", "carol"})

    def test_prefill_is_kept_once(self):
        self.assertEqual(with_prefill("@alice have you seen it", "@alice "), "@alice have you seen it")
        self.assertEqual(with_prefill("have you seen it", "@alice "), "@alice have you seen it")
        self.assertEqual(with_prefill("plain reply", ""), "plain reply")

    def test_reply_keeps_only_the_personas_own_clean_lines(self):
        raw = "bob: sure thing\ncheck [link example.com] out\nsure thing\nalice: no way"
        self.assertEqual(clean_reply(raw, "bob", {"bob", "alice"}), "sure thing\ncheck out")


class Channel:
    def get_channel(self, channel_id):
        return channel_id


def bots(length=5, seed=0):
    people = People([Persona(p, token="t") for p in ("bob", "alice", "carol")], {})
    args = argparse.Namespace(llm_url="", channel=[], downweight=[], converse_length=length)
    b = Bots(people, {}, args, random.Random(seed))
    b.clients = {p: Channel() for p in b.personas}
    return b


@patch("understudy.bot.asyncio.sleep", new=AsyncMock())
class Conversation(unittest.IsolatedAsyncioTestCase):
    async def test_whoever_is_named_answers_next_until_someone_has_nothing(self):
        b = bots(seed=1)  # bob opens
        replies = {"bob": "over to alice", "alice": "carol, you?", "carol": ""}
        spoken = []

        async def speak(channel, persona, prefill="", since=None):
            spoken.append((persona, prefill))
            return replies[persona]

        with patch.object(b, "speak", side_effect=speak):
            await b.converse(1)
        self.assertEqual([p for p, _ in spoken], ["bob", "alice", "carol"])
        self.assertTrue(spoken[0][1].startswith("@"))
        self.assertEqual([prefill for _, prefill in spoken[1:]], ["", ""])

    async def test_a_person_writing_stops_it(self):
        b = bots()
        calls = 0

        async def speak(channel, persona, prefill="", since=None):
            nonlocal calls
            calls += 1
            b.last_person[1] = datetime.now(UTC) + timedelta(seconds=1)
            return "anyone else?"

        with patch.object(b, "speak", side_effect=speak):
            await b.converse(1)
        self.assertEqual(calls, 1)

    async def test_it_stops_at_its_length(self):
        b = bots(length=3)
        with patch.object(b, "speak", new=AsyncMock(return_value="go on")) as speak:
            await b.converse(1)
        self.assertEqual(speak.await_count, 3)


if __name__ == "__main__":
    unittest.main()
