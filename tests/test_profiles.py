import tempfile
import unittest
from pathlib import Path

from understudy.profiles import read_profiles, system_prompt


class Profiles(unittest.TestCase):
    def test_reads_blocks_and_skips_comments(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
            fh.write("# header\n[Bob]\nTwo kids.  # aside\nPlays PUBG.\n\n[empty]\n[alice]\nTypo queen.\n")
        profiles = read_profiles(Path(fh.name))
        Path(fh.name).unlink()
        self.assertEqual(profiles, {"bob": "Two kids.\nPlays PUBG.", "alice": "Typo queen."})

    def test_missing_file_means_no_profiles(self):
        self.assertEqual(read_profiles(Path("/nonexistent/profiles.txt")), {})

    def test_system_prompt_matches_training_without_a_profile(self):
        self.assertEqual(system_prompt("carol", "general", {"bob": "x"}),
                         "Reply as `carol` in #general.")
        self.assertEqual(system_prompt("Bob", "pubg", {"bob": "Two kids.\nPlays PUBG."}),
                         "Reply as `Bob` in #pubg.\nAbout Bob: Two kids. Plays PUBG.")


if __name__ == "__main__":
    unittest.main()
