#!/usr/bin/env python3
"""QLoRA fine-tune of one shared adapter across all personas.

    python scripts/train.py --data data/ --out outputs/adapter

Needs CUDA (run under WSL2 on the training box). Writes the LoRA adapter only.
"""

from __future__ import annotations

# unsloth patches transformers/trl on import, so it has to come first
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only

import argparse
from pathlib import Path

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

# Llama 3 header tokens; loss is masked to everything after the assistant header
INSTRUCTION_PART = "<|start_header_id|>user<|end_header_id|>\n\n"
RESPONSE_PART = "<|start_header_id|>assistant<|end_header_id|>\n\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("outputs/adapter"))
    ap.add_argument("--model", default="AI-Sweden-Models/Llama-3-8B-instruct")
    ap.add_argument("--max-seq-length", type=int, default=1024)
    # low rank on purpose: high rank mostly buys verbatim memorization
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model, tok = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    ds = load_dataset("json", data_files={
        "train": str(args.data / "train.jsonl"),
        "val": str(args.data / "val.jsonl"),
    })
    ds = ds.map(lambda r: {"text": tok.apply_chat_template(r["messages"], tokenize=False)},
                remove_columns=["messages"])

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        args=SFTConfig(
            output_dir=str(args.out.parent / "checkpoints"),
            dataset_text_field="text",
            max_seq_length=args.max_seq_length,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            optim="adamw_8bit",
            bf16=True,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=3,
            seed=args.seed,
            report_to="none",
        ),
    )
    # without this the model also learns to write the context, i.e. everyone else's voice
    trainer = train_on_responses_only(trainer, instruction_part=INSTRUCTION_PART,
                                      response_part=RESPONSE_PART)
    trainer.train()

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"adapter -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
