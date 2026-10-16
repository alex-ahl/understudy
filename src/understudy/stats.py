"""Token lengths of the dataset under the model's chat template.

    understudy stats private/data/train.jsonl --model AI-Sweden-Models/Llama-3-8B-instruct

Pick max_seq_length just above p95; longer examples get truncated in training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from understudy.train import MODEL


def percentile(sorted_vals: list[int], q: float) -> int:
    return sorted_vals[min(len(sorted_vals) - 1, int(len(sorted_vals) * q))]


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="understudy stats", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("data", type=Path, nargs="?", default=Path("private/data/train.jsonl"))
    ap.add_argument("--model", default=MODEL)
    ap.add_argument(
        "--max-seq-length", type=int, default=1024, help="warn when p95 is above it (default 1024, as in train)"
    )
    return ap


def run(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    rows = [json.loads(line) for line in args.data.open(encoding="utf-8")]
    # measured through the template, not raw text: it wraps every turn in header tokens.
    # render then tokenize: tokenize=True returns a dict in newer transformers, and len() counts its keys
    lengths = sorted(
        len(tok(tok.apply_chat_template(r["messages"], tokenize=False), add_special_tokens=False).input_ids)
        for r in rows
    )

    print(f"examples  {len(lengths)}")
    for label, q in (("p50", 0.50), ("p90", 0.90), ("p95", 0.95), ("p99", 0.99)):
        print(f"{label:<9} {percentile(lengths, q)}")
    print(f"max       {lengths[-1]}")
    for limit in (512, 768, 1024, 1536, 2048):
        over = sum(n > limit for n in lengths)
        print(f"  > {limit:<5} {over:>6}  ({over / len(lengths):.1%})")
    p95 = percentile(lengths, 0.95)
    if p95 > args.max_seq_length:
        print(
            f"\np95 {p95} is over max_seq_length {args.max_seq_length}: training truncates over 5% of examples",
            file=sys.stderr,
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
