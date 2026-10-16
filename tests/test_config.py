import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from understudy import cli
from understudy.config import ConfigError, check, defaults, load

ROOT = Path(__file__).resolve().parent.parent


def parsed(config, command, *argv):
    parser = cli.module(command).parser()
    parser.set_defaults(**defaults(config, command, parser))
    return parser.parse_args(argv)


def parsers():
    return {name: cli.module(name).parser() for name in cli.COMMANDS}


class Config(unittest.TestCase):
    def test_section_beats_top_level_and_the_command_line_beats_both(self):
        config = {"seed": 1, "train": {"seed": 2}}
        self.assertEqual(parsed(config, "dataset").seed, 1)
        self.assertEqual(parsed(config, "train").seed, 2)
        self.assertEqual(parsed(config, "train", "--seed", "3").seed, 3)

    def test_top_level_keys_go_only_to_commands_with_that_option(self):
        config = {"model": "some/model", "no_scrub": True}
        self.assertEqual(parsed(config, "train").model, "some/model")
        self.assertTrue(parsed(config, "dataset").no_scrub)
        self.assertFalse(hasattr(parsed(config, "train"), "no_scrub"))

    def test_paths_are_paths(self):
        self.assertEqual(parsed({"dataset": {"input": "exports"}}, "dataset").input, Path("exports"))

    def test_lists_extend_with_the_command_line(self):
        args = parsed({"dataset": {"persona": ["bob"]}}, "dataset", "--persona", "alice")
        self.assertEqual(args.persona, ["bob", "alice"])

    def test_people_tables_go_to_the_commands_that_read_them(self):
        config = {"persona": {"bob": {"names": ["robert"]}}, "aliases": {"carl_xx": "carl"}}
        self.assertEqual(parsed(config, "dataset").people, config)
        self.assertEqual(parsed(config, "bot").people, config)
        self.assertFalse(hasattr(parsed(config, "train"), "people"))

    def test_unknown_keys_and_sections_are_errors(self):
        check(
            {
                "seed": 1,
                "train": {"epochs": 3},
                "persona": {"bob": {"names": ["robert"]}},
                "aliases": {"carl_xx": "carl"},
            },
            parsers(),
        )
        for bad in (
            {"train": {"epoch": 3}},
            {"trian": {}},
            {"no_such_option": 1},
            {"persona": {"bob": {"name": ["robert"]}}},
            {"persona": {"bob": {"names": "robert"}}},
            {"persona": {"bob": "robert"}},
            {"aliases": {"carl_xx": 1}},
        ):
            with self.subTest(bad), self.assertRaises(ConfigError):
                check(bad, parsers())


class ConfigFile(unittest.TestCase):
    def test_the_example_config_is_valid(self):
        check(load(ROOT / "understudy.example.toml"), parsers())

    def test_dataset_runs_from_the_private_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "private").mkdir()
            (tmp / "export.json").write_text(
                json.dumps(
                    {
                        "channel": {"name": "general"},
                        "messages": [
                            {
                                "id": str(i),
                                "type": "Default",
                                "timestamp": f"2026-01-01T12:0{i}:00+00:00",
                                "author": {"id": a, "name": a, "isBot": False},
                                "content": text,
                                "attachments": [],
                                "mentions": [],
                            }
                            for i, (a, text) in enumerate(
                                [
                                    ("carol", "something to reply to"),
                                    ("robert", "something else as well"),
                                    ("bob", "bob has thoughts on it"),
                                ]
                            )
                        ],
                    }
                )
            )
            (tmp / "private" / "understudy.toml").write_text(
                '[persona.bob]\n[aliases]\nrobert = "pat"\n'
                '[dataset]\ninput = "export.json"\nout = "data"\nno_scrub = true\n'
            )
            proc = subprocess.run(
                [sys.executable, "-m", "understudy", "dataset"], capture_output=True, text=True, cwd=tmp, check=False
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            rows = [
                json.loads(line)
                for name in ("train", "val")
                if (tmp / "data" / f"{name}.jsonl").exists()
                for line in (tmp / "data" / f"{name}.jsonl").read_text().splitlines()
            ]
        [row] = rows
        self.assertIn("pat: something else", row["messages"][1]["content"])

    def test_a_bad_config_stops_before_running(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "understudy.toml"
            path.write_text("[train]\nepoch = 3\n")
            proc = subprocess.run(
                [sys.executable, "-m", "understudy", "--config", path, "dataset"],
                capture_output=True,
                text=True,
                cwd=tmp,
                check=False,
            )
        self.assertEqual(proc.returncode, 2)
        self.assertIn("epoch", proc.stderr)


class Pipeline(unittest.TestCase):
    def test_stage_ranges(self):
        self.assertEqual(cli.stages(None, None), ["dataset", "stats", "train", "convert"])
        self.assertEqual(cli.stages("stats", "train"), ["stats", "train"])
        self.assertEqual(cli.stages("convert", None), ["convert"])
        self.assertEqual(cli.stages("train", "stats"), [])

    def test_stops_at_the_first_failing_stage(self):
        ran = []

        def stage(name):
            ran.append(name)
            return 1 if name == "stats" else 0

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = cli.pipeline(["dataset", "stats", "train"], stage)
        self.assertEqual(code, 1)
        self.assertEqual(ran, ["dataset", "stats"])


if __name__ == "__main__":
    unittest.main()
