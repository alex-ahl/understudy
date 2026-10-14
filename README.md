# understudy

A LoRA fine-tune that replies in the voice of specific people from a Discord
server, and Discord bots that post as them. One adapter covers every persona;
the system prompt names the speaker: ``Reply as `bob` in #general.``

Training needs an NVIDIA GPU (WSL2 on Windows works). Serving runs anywhere
llama.cpp does.

## Before you start

- Get consent from everyone being imitated.
- Read history with a bot token only.
- Label bot output visibly, e.g. a distinct avatar and name per persona.

## Settings

Everything private lives in `private/` (gitignored): the settings file, the
export, the dataset and the trained adapter. Commands run from the repo root.

```sh
mkdir -p private && cp understudy.example.toml private/understudy.toml
```

`private/understudy.toml` is the only file to edit; `understudy.example.toml`
explains every setting. It holds:

- `[persona.NAME]`, one per person to imitate: their Discord `names`, a
  `profile` in their own voice, and their bot's `token`.
- `[aliases]`: everyone else to rename.
- Settings for each command, all optional. Keys are the long options with `_`
  for `-`; flags on the command line win over the file. An unknown key stops
  the command.

`understudy --config FILE <command>` reads another file. With
`private/understudy.toml` in place, `python -m unittest discover -s tests`
fails if a tracked file contains a name, profile line or token from it.

## Pipeline

```sh
understudy pipeline                         # dataset -> stats -> train -> convert
understudy pipeline --to train              # stop after training
understudy pipeline --from convert          # start from an existing adapter
```

Each stage runs with the settings file's values and stops the pipeline when it
fails. Stages run wherever their requirements are installed; copy
`private/data/` or `private/outputs/` between machines to split the pipeline
across them. `serve` and `bot`
run until stopped and are started separately.

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
python3 -m venv .venv && source .venv/bin/activate   # Python 3.11+
pip install -e '.[dataset]'
understudy dataset              # private/exports/ -> private/data/
```

The `[persona.*]` tables are who to imitate. Each persona's `names`, and every
name in `[aliases]`, is replaced everywhere: authors, mentions and plain text.

```toml
[persona.bob]
names = ["robert_1987", "Robban"]

[aliases]
carl_xx = "carl"
```

- Case, accents and stray marks like `´` are ignored.
- Names match whole words, plus a possessive `s`: `roberts` → `bob`. A name
  inside a longer word stays as it is; list compounds you want replaced.
- Renaming one name to two different pseudonyms is an error.

`--persona NAME` and `--alias REAL=FAKE`, each repeatable, add more on the command line.

### Scrubbing

[Presidio](https://github.com/microsoft/presidio) with a Swedish spaCy model and
[GLiNER](https://github.com/urchade/GLiNER) replaces, in the dataset only:

- links, down to their domain: `[link youtube.com]`
- emails, phone numbers, addresses and passwords: `[email]`, `[number]`,
  `[address]`, `[password]`
- towns and other places: `[place]`
- names not renamed by the config: `[name]`

Organisations are listed after the run, not replaced; alias the ones that identify
someone, like an employer. The models download on the first run (about 1 GB).
`--no-scrub` turns scrubbing off. The bots don't scrub.

A persona needs ~500 examples. Below that, add channels or drop the persona.
`--max-per-persona N` samples larger personas down to N.

### Shaping

- Replies containing a link are dropped.
- `downweight` lists case-insensitive regexes. Matching replies, and matching
  context lines other than the one replied to, are kept at 20%.
- `private/data/voices.json` holds each persona's plain one-liners from the
  train split. Every example gets up to three of them, then some of its
  persona's profile lines, as that persona's messages a few turns from the end.
  The bots put three of them and the whole profile in the same place.

### Persona profiles

A persona's `profile` is two or three lines in their own voice. As described
under Shaping, they go into the conversation as that persona's earlier
messages, the last one closest to the reply.

## 3. Set up the training box

```sh
nvidia-smi                      # must work; under WSL2, do not install a driver inside WSL
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.11 && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install -e '.[train]'
```

Under WSL2, keep the repo and data under `~/`, not `/mnt/c/`.

## 4. Train

```sh
understudy stats                          # token lengths of private/data/train.jsonl
understudy train --max-seq-length 1024    # just above p95
```

`stats` warns when p95 is above `--max-seq-length`. The adapter lands in
`private/outputs/adapter/`. `--model` picks another base model; the
response masking in `understudy train` assumes the Llama 3 chat format.

## 5. Serve

With the adapter in `private/outputs/adapter/`:

```sh
brew install llama.cpp          # or build it; llama-server must be on PATH
git clone --depth 1 https://github.com/ggml-org/llama.cpp ../llama.cpp
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -r ../llama.cpp/requirements/requirements-convert_lora_to_gguf.txt

understudy convert --llama-cpp ../llama.cpp   # -> private/outputs/adapter.gguf
understudy serve                              # downloads the base model on first run, serves on :8080
```

`--base-gguf FILE` serves a local base model. `--lora-scale` above 1.0
exaggerates the personas; `--context` and `--port` set llama-server's `-c` and
`--port`.

## 6. Discord bots

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
   source .venv/bin/activate
   pip install -e '.[bot]'
   understudy bot
   ```

A persona replies to the channel's recent conversation when someone @-mentions
it or writes its name. `--channel NAME`, repeatable, limits where;
`--llm-url` points at another server. They default to the `CHANNELS`
(comma-separated) and `LLM_URL` environment variables.

To have the personas talk among themselves:

```toml
[bot]
converse_every = 60         # minutes
converse_length = 20        # messages
converse_channel = "general"
```

One persona opens by addressing another; whoever a message names answers next,
otherwise another persona at random. A person writing in the channel ends the
conversation. Needs two or more personas with a token.

Run `python -m unittest discover -s tests` after changing the code.
