# understudy

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Llama 3](https://img.shields.io/badge/base-Llama%203%208B-0467DF?logo=meta&logoColor=white)](https://huggingface.co/AI-Sweden-Models/Llama-3-8B-instruct)
[![QLoRA with Unsloth](https://img.shields.io/badge/QLoRA-Unsloth-8A2BE2)](https://github.com/unslothai/unsloth)
[![llama.cpp](https://img.shields.io/badge/serve-llama.cpp-000000)](https://github.com/ggml-org/llama.cpp)
[![discord.py](https://img.shields.io/badge/bots-discord.py-5865F2?logo=discord&logoColor=white)](https://github.com/Rapptz/discord.py)
[![Presidio](https://img.shields.io/badge/PII-Presidio%20%2B%20GLiNER-0078D4)](https://github.com/microsoft/presidio)

**Your Discord friends, as bots that talk like them.**

understudy fine-tunes one small LoRA adapter on a server's chat history and runs
a Discord bot per person that answers in their voice: their slang, their
running jokes, their opinions about phones.

```
Discord export ──► dataset ──► LoRA fine-tune ──► llama-server ──► one bot per persona
  (DCE JSON)       scrubbed,      one adapter,        GGUF, runs        @-mention them, or let
                   renamed        every persona       on a laptop       them talk among themselves
```

- **One adapter, many people.** The system prompt names the speaker:
  ``Reply as `bob` in #general.``
- **Private by default.** Real names become pseudonyms; links, emails, phone
  numbers, addresses and stray names are scrubbed before training. Everything
  private lives in one gitignored folder.
- **One settings file.** People, profiles, bot tokens and every command's
  settings live in `private/understudy.toml`.
- **Runs where you have the hardware.** Train on an NVIDIA GPU (WSL2 works),
  serve anywhere llama.cpp runs.

## Before you start

- Get consent from everyone being imitated.
- Read history with a bot token only.
- Label bot output visibly, e.g. a distinct avatar and name per persona.

## Requirements

| Stage | Needs |
|---|---|
| dataset | Python 3.11+, `pip install -e '.[dataset]'`, ~1 GB of models on first run |
| train | NVIDIA GPU, CUDA build of PyTorch, `pip install -e '.[train]'` |
| convert, serve | a llama.cpp checkout and `llama-server` |
| bot | `pip install -e '.[bot]'`, a Discord bot account per persona |

## Quick start

```sh
python3 -m venv .venv && source .venv/bin/activate
mkdir -p private && cp understudy.example.toml private/understudy.toml
# edit private/understudy.toml, export the server into private/exports/ (step 1)
understudy pipeline          # dataset -> stats -> train -> convert
understudy serve             # in one terminal
understudy bot               # in another
```

## Settings

Everything private lives in `private/` (gitignored): the settings file, the
export, the dataset and the trained adapter. Commands run from the repo root.

`private/understudy.toml` is the only file to edit; `understudy.example.toml`
explains every setting.

```toml
[persona.bob]                          # one table per person to imitate
names = ["robert_1987", "Robban"]      # how they appear in Discord
profile = ["gaming again tonight, anyone in?"]
token = "..."                          # their bot's token

[aliases]                              # everyone else to rename
carl_xx = "carl"

[train]                                # any command's options, all optional
epochs = 3
```

- Option keys are the long options with `_` for `-`; flags on the command line
  win over the file. An unknown key stops the command.
- `understudy --config FILE <command>` reads another file.
- With `private/understudy.toml` in place, `python -m unittest discover -s tests`
  fails if a tracked file contains a name, profile line or token from it.

## 1. Export

1. Discord Developer Portal → New Application → Bot → enable **Message Content
   Intent**.
2. OAuth2 → URL Generator: scope `bot`, permissions `View Channels` and
   `Read Message History`. Invite it to the server.
3. Export:

   ```sh
   docker run --rm -v "$PWD/private/exports:/out" tyrrrz/discordchatexporter:stable \
     exportguild -t "$BOT_TOKEN" -g "$GUILD_ID" -f Json -o /out/
   ```

## 2. Build the dataset

```sh
pip install -e '.[dataset]'
understudy dataset              # private/exports/ -> private/data/
```

### Renaming

Each persona's `names`, and every name in `[aliases]`, is replaced everywhere:
authors, mentions and plain text.

- Case, accents and stray marks like `´` are ignored.
- Names match whole words, plus a possessive `s`: `roberts` → `bob`. A name
  inside a longer word stays as it is; list compounds you want replaced.
- Renaming one name to two different pseudonyms is an error.
- `--persona NAME` and `--alias REAL=FAKE`, each repeatable, add more on the
  command line.

### Scrubbing

[Presidio](https://github.com/microsoft/presidio) with a Swedish spaCy model and
[GLiNER](https://github.com/urchade/GLiNER) replace, in the dataset only:

| Found | Becomes |
|---|---|
| links | `[link youtube.com]` |
| emails, phone numbers, addresses, passwords | `[email]`, `[number]`, `[address]`, `[password]` |
| towns and other places | `[place]` |
| names not renamed by the config | `[name]` |

Organisations are listed after the run, not replaced; alias the ones that
identify someone, like an employer. `--no-scrub` turns scrubbing off. The bots
don't scrub.

### Shaping

- Replies containing a link are dropped.
- `downweight` lists case-insensitive regexes. Matching replies, and matching
  context lines other than the one replied to, are kept at 20%.
- `max_per_persona` samples larger personas down. A persona needs ~500
  examples; below that, add channels or drop the persona.
- `private/data/voices.json` holds each persona's plain one-liners from the
  train split. Every example gets up to three of them, then some of its
  persona's `profile` lines, as that persona's messages a few turns from the
  end, the last profile line closest to the reply. The bots put three of them
  and the whole profile in the same place.

## 3. Train

Set up the training box:

```sh
nvidia-smi                      # must work; under WSL2, do not install a driver inside WSL
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.11 && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install -e '.[train]'
```

Under WSL2, keep the repo and data under `~/`, not `/mnt/c/`.

```sh
understudy stats                          # token lengths of private/data/train.jsonl
understudy train --max-seq-length 1024    # just above p95
```

`stats` warns when p95 is above `--max-seq-length`. The adapter lands in
`private/outputs/adapter/`. `--model` picks another base model; the response
masking in `understudy train` assumes the Llama 3 chat format.

## 4. Serve

```sh
brew install llama.cpp          # or build it; llama-server must be on PATH
git clone --depth 1 https://github.com/ggml-org/llama.cpp ../llama.cpp
pip install -e .
pip install -r ../llama.cpp/requirements/requirements-convert_lora_to_gguf.txt

understudy convert --llama-cpp ../llama.cpp   # -> private/outputs/adapter.gguf
understudy serve                              # downloads the base model on first run, serves on :8080
```

`--base-gguf FILE` serves a local base model. `--lora-scale` above 1.0
exaggerates the personas; `--context` and `--port` set llama-server's `-c` and
`--port`.

## 5. Discord bots

Each persona is its own bot account, running next to `llama-server`.

1. For each persona, in the Developer Portal: **New Application** named after
   the persona → **Bot**: set the avatar, enable **Message Content Intent**,
   copy the token.
2. OAuth2 → URL Generator for each app: scope `bot`, permissions
   `View Channels`, `Read Message History`, `Send Messages`. Open the URL and add
   the bot to the server.
3. Put each token in its persona's table: `token = "..."`.
4. Start them:

   ```sh
   pip install -e '.[bot]'
   understudy bot
   ```

A persona replies to the channel's recent conversation when someone @-mentions
it or writes its name. `--channel NAME`, repeatable, limits where; `--llm-url`
points at another server. They default to the `CHANNELS` (comma-separated) and
`LLM_URL` environment variables.

### Conversations among the personas

```toml
[bot]
converse_every = 60         # minutes
converse_length = 20        # messages
converse_channel = "general"
```

One persona opens by addressing another; whoever a message names answers next,
otherwise another persona at random. A person writing in the channel ends the
conversation. Needs two or more personas with a token.

## Pipeline

```sh
understudy pipeline                         # dataset -> stats -> train -> convert
understudy pipeline --to train              # stop after training
understudy pipeline --from convert          # start from an existing adapter
```

Each stage runs with the settings file's values and stops the pipeline when it
fails. Stages run wherever their requirements are installed; copy
`private/data/` or `private/outputs/` between machines to split the pipeline
across them. `serve` and `bot` run until stopped and are started separately.

## Development

```sh
pip install -e '.[dataset]'
python -m unittest discover -s tests
```
