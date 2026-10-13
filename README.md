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
make dataset
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

`PERSONAS="..."` and `ALIASES="REAL=FAKE ..."` add more on the command line.
Emails, phone numbers, invites, links, passwords and Swedish street addresses are
scrubbed.

A persona needs ~500 examples. Below that, add channels or drop the persona.

### Persona profiles

`profiles.txt` (gitignored) gives each persona two or three lines in their own
voice; start from `profiles.example.txt`. The bots put them into the conversation
as that persona's earlier messages, the last one closest to the reply.
`make dataset` also puts them in the training system prompt.

## 3. Set up the training box

```sh
nvidia-smi                      # must work; under WSL2, do not install a driver inside WSL
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.11 && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install -r requirements-train.txt
```

Under WSL2, keep the repo and data under `~/`, not `/mnt/c/`.

## 4. Train

```sh
make stats                      # set MAX_SEQ just above p95
make train MAX_SEQ=1024
```

The adapter lands in `outputs/adapter/`. `MODEL=` picks another base model; the
response masking in `scripts/train.py` assumes the Llama 3 chat format.

## 5. Serve

With the adapter in `outputs/adapter/`:

```sh
brew install llama.cpp          # or build it; llama-server must be on PATH
git clone --depth 1 https://github.com/ggml-org/llama.cpp ../llama.cpp
python3 -m venv .venv && source .venv/bin/activate
pip install -r ../llama.cpp/requirements/requirements-convert_lora_to_gguf.txt

make lora-gguf                  # outputs/adapter.gguf
make serve                      # downloads the base model on first run, then serves on :8080
```

`LORA_SCALE` (default 1.0) above 1 exaggerates the personas.

## 6. Discord bots

Each persona is its own bot account, running next to `make serve`.

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
   pip install -r requirements-bot.txt
   make bot
   ```

A persona replies to the channel's recent conversation when someone @-mentions
it or writes its name. `CHANNELS` (comma-separated names) limits where;
`LLM_URL` points at another server.

Run `make test` after changing the scripts.
