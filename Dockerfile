# syntax=docker/dockerfile:1
# understudy on top of llama.cpp's own images, which bring llama-server and the
# LoRA converter (in /app, its requirements in the system Python). compose.yaml
# builds it twice:
#   cpu   BASE=llama.cpp:full       dataset, stats, convert, serve on CPU, bot
#   cuda  BASE=llama.cpp:full-cuda  train, convert, serve on an NVIDIA GPU, bot

ARG BASE=ghcr.io/ggml-org/llama.cpp:full
FROM ${BASE}

# extras to install, the torch build to take them with, and whether to pin the
# dataset stack to docker/constraints.txt (a set that passes the tests)
ARG EXTRAS=dataset,bot
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG PIN=true
ARG SPACY_MODEL=https://github.com/explosion/spacy-models/releases/download/sv_core_news_sm-3.8.0/sv_core_news_sm-3.8.0-py3-none-any.whl

# build-essential: Triton compiles small launchers when Unsloth trains
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3-venv build-essential \
 && rm -rf /var/lib/apt/lists/*

# understudy gets its own environment: the converter in the system Python wants an
# older transformers than the scrubber does
RUN python3 -m venv /venv
ENV PATH=/venv/bin:/app:$PATH \
    HF_HOME=/cache/huggingface

WORKDIR /work
COPY pyproject.toml README.md ./
COPY docker/constraints.txt docker/
COPY src src
RUN pip install --no-cache-dir --extra-index-url "$TORCH_INDEX" \
      $([ "$PIN" = true ] && echo "-c docker/constraints.txt") \
      -e ".[$EXTRAS]" protobuf \
 && case ",$EXTRAS," in *,dataset,*) pip install --no-cache-dir "$SPACY_MODEL";; esac
COPY tests tests

ENTRYPOINT ["understudy"]
