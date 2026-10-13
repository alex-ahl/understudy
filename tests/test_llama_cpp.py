import sys
import unittest
from pathlib import Path

from understudy import convert, serve


class Convert(unittest.TestCase):
    def test_command(self):
        args = convert.parser().parse_args(["--llama-cpp", "/src/llama.cpp", "--model", "some/model"])
        self.assertEqual(convert.command(args), [
            sys.executable, "/src/llama.cpp/convert_lora_to_gguf.py", "outputs/adapter",
            "--base-model-id", "some/model", "--outfile", "outputs/adapter.gguf"])


class Serve(unittest.TestCase):
    def test_downloads_the_base_by_default(self):
        args = serve.parser().parse_args(["--lora-scale", "1.1"])
        self.assertEqual(serve.command(args), [
            "llama-server", "-hf", serve.BASE_HF, "--lora-scaled", "outputs/adapter.gguf:1.1",
            "-c", "4096", "--port", "8080"])

    def test_a_local_base_wins(self):
        args = serve.parser().parse_args(["--base-gguf", "/models/base.gguf"])
        self.assertEqual(serve.command(args)[1:3], ["-m", str(Path("/models/base.gguf"))])


if __name__ == "__main__":
    unittest.main()
