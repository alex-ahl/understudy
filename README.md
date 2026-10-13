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

Every command reads `understudy.toml` (gitignored) from the working directory,
or the file given with `understudy --config FILE <command>`. Start from
`understudy.example.toml`.

- Keys are the commands' long options with `_` for `-`: `max_seq_length = 1536`
  for `--max-seq-length 1536`. `understudy <command> -h` lists them.
- A top-level key applies to every command with that option; a `[command]`
  section applies to that command and wins over the top level.
- Flags on the command line win over the file.
- Relative paths resolve against the file's directory.
- An unknown key or section stops the command.

Keep private files in the repo root, where `.gitignore` covers them:
`understudy.toml`, `aliases.txt`, `profiles.txt`, `bots.txt`, `exports/`,
`data/`, `outputs/`. With them in place,
`python -m unittest discover -s tests` fails if a tracked file contains a name,
profile line or token from them.

## Pipeline

```sh
understudy pipeline                         # dataset -> stats -> train -> convert
understudy pipeline --to train              # stop after training
understudy pipeline --from convert          # start from an existing adapter
```

Each stage runs with the settings file's values and stops the pipeline when it
fails. Stages run wherever their requirements are installed; copy `data/` or
`outputs/` between machines to split the pipeline across them. `serve` and `bot`
run until stopped and are started separately.

## 1. Export

1. Discord Developer Portal → New Application → Bot → enable **Message Content
   Intent**.
2. OAuth2 → URL Generator: scope `bot`, permissions `View Channels` and
   `Read Message History`. Invite it to the server.
3. Export:

   ```sh
   docker run --rm -v "$PWD/exports:/out" tyrrrz/discordchatexporter:stable \
     exportguild -t "$BOT_TOKEN" -g "$GUILD_ID" -f Json -o /out/
   ```

## 2. Build the dataset

```sh
python3 -m venv .venv && source .venv/bin/activate   # Python 3.11+
pip install -e '.[dataset]'
understudy dataset              # exports/ -> data/train.jsonl, data/val.jsonl
```

`aliases.txt` lists who to imitate and which names to replace. One entry per
line, `#` starts a comment:

```
*robert_1987=bob      # persona, aliased
robert=bob            # extra spellings of the same person
*alice                # persona, real name kept
carl_xx=carl          # not a persona, aliased in context
```

- `*` marks a persona.
- `REAL=FAKE` replaces the name everywhere: authors, mentions, and plain text.
  Any word containing `REAL` becomes `FAKE` (`roberts` → `bob`), so short names
  also catch unrelated words. Case, accents, and stray marks like `´` are ignored.
- `"REAL"=FAKE` (quoted) matches the whole word only, plus a possessive `s`.

`--persona NAME` and `--alias REAL=FAKE`, each repeatable, add more on the command line.

### Scrubbing

[Presidio](https://github.com/microsoft/presidio) with a Swedish spaCy model and
[GLiNER](https://github.com/urchade/GLiNER) replaces, in the dataset only:

- links, down to their domain: `[link youtube.com]`
- emails, phone numbers, addresses and passwords: `[email]`, `[number]`,
  `[address]`, `[password]`
- towns and other places: `[place]`
- names not in `aliases.txt`: `[name]`

Organisations are listed after the run, not replaced; alias the ones that identify
someone, like an employer. The models download on the first run (about 1 GB).
`--no-scrub` turns scrubbing off. The bots don't scrub.

A persona needs ~500 examples. Below that, add channels or drop the persona.

### Persona profiles

`profiles.txt` (gitignored) gives each persona two or three lines in their own
voice; start from `profiles.example.txt`. The bots put them into the conversation
as that persona's earlier messages, the last one closest to the reply.
`understudy dataset` also puts them in the training system prompt.

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
understudy stats                          # token lengths of data/train.jsonl
understudy train --max-seq-length 1024    # just above p95
```

`stats` warns when p95 is above `--max-seq-length`. The adapter lands in
`outputs/adapter/`. `--model` picks another base model; the
response masking in `understudy train` assumes the Llama 3 chat format.

## 5. Serve

With the adapter in `outputs/adapter/`:

```sh
brew install llama.cpp          # or build it; llama-server must be on PATH
git clone --depth 1 https://github.com/ggml-org/llama.cpp ../llama.cpp
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -r ../llama.cpp/requirements/requirements-convert_lora_to_gguf.txt

understudy convert --llama-cpp ../llama.cpp   # -> outputs/adapter.gguf
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
3. Put the tokens in `bots.txt` (gitignored), one `persona=token` per line.
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

Run `python -m unittest discover -s tests` after changing the code.
