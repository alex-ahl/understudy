import tempfile
import unittest
from pathlib import Path

from understudy.profiles import build_messages, read_profiles, system_prompt


class Profiles(unittest.TestCase):
    def test_reads_blocks_and_skips_comments(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
            fh.write("# header\n[Bob]\nTwo kids.  # aside\nPlays PUBG.\n\n[empty]\n[alice]\nTypo queen.\n")
        profiles = read_profiles(Path(fh.name))
        Path(fh.name).unlink()
        self.assertEqual(profiles, {"bob": "Two kids.\nPlays PUBG.", "alice": "Typo queen."})

    def test_missing_file_means_no_profiles(self):
        self.assertEqual(read_profiles(Path("/nonexistent/profiles.txt")), {})

    def test_system_prompt_names_the_persona_and_channel(self):
        self.assertEqual(system_prompt("Bob", "pubg"), "Reply as `Bob` in #pubg.")

    def test_own_lines_go_in_near_the_end_as_the_personas_messages(self):
        turns = [f"p{i}: line {i}" for i in range(6)]
        system, user = build_messages("bob", "general", turns, ["said this once", "Likes trains."])
        self.assertEqual(user["content"].splitlines(),
                         ["p0: line 0", "p1: line 1", "p2: line 2",
                          "bob: said this once", "bob: Likes trains.",
                          "p3: line 3", "p4: line 4", "p5: line 5"])
        self.assertEqual(system["content"], "Reply as `bob` in #general.")

    def test_a_short_context_gets_own_lines_first(self):
        _, user = build_messages("bob", "general", ["carol: hi"], ["said this once"])
        self.assertEqual(user["content"], "bob: said this once\ncarol: hi")


if __name__ == "__main__":
    unittest.main()
