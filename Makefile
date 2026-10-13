MODEL      ?= AI-Sweden-Models/Llama-3-8B-instruct
MAX_SEQ    ?= 1024
ADAPTER    ?= outputs/adapter
PERSONAS   ?=
ALIASES    ?=
ALIAS_FILE ?= aliases.txt

PY := python3

.PHONY: dataset stats train lora-gguf serve bot test

# make dataset PERSONAS="bob alice" ALIASES="robert=bob"
dataset:
	$(PY) scripts/discord_to_sft.py exports/ -o data/ \
		$(foreach p,$(PERSONAS),--persona $(p)) \
		$(foreach a,$(ALIASES),--alias $(a)) \
		$(if $(wildcard $(ALIAS_FILE)),--aliases $(ALIAS_FILE)) \
		$(if $(wildcard profiles.txt),--profiles profiles.txt)

stats:
	$(PY) scripts/token_stats.py data/train.jsonl --model $(MODEL)

train:
	$(PY) scripts/train.py --data data/ --out $(ADAPTER) \
		--model $(MODEL) --max-seq-length $(MAX_SEQ)

# llama.cpp runs the adapter on top of a quantized base, no merge needed
# a Hugging Face GGUF repo, or a local .gguf path
BASE_GGUF  ?= AI-Sweden-Models/Llama-3-8B-instruct-Q4_K_M-gguf
LLAMA_CPP  ?= ../llama.cpp
LORA_SCALE ?= 1.0  # >1 exaggerates the personas; gibberish somewhere past 2

lora-gguf:
	$(PY) $(LLAMA_CPP)/convert_lora_to_gguf.py $(ADAPTER) \
		--base-model-id $(MODEL) --outfile outputs/adapter.gguf

serve:
	llama-server $(if $(wildcard $(BASE_GGUF)),-m,-hf) $(BASE_GGUF) --lora-scaled outputs/adapter.gguf:$(LORA_SCALE) -c 4096 --port 8080

# tokens in bots.txt; needs make serve running
bot:
	$(PY) scripts/bot.py

test:
	$(PY) -m unittest discover -s tests
