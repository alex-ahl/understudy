# syntax=docker/dockerfile:1
# One image per job, so each machine pulls only what it runs (compose.yaml picks
# the target):
#   cpu         dataset, stats, bot          python slim + CPU torch
#   train       train                        python slim + CUDA torch + Unsloth
#   convert     convert                      llama.cpp's full image (the converter)
#   serve       serve on CPU                 llama.cpp's server image
#   serve-cuda  serve on an NVIDIA GPU       llama.cpp's CUDA server image
# understudy's own code needs only the standard library, so on llama.cpp's images
# it runs on their system Python without pip.

ARG LLAMA_CPP=ghcr.io/ggml-org/llama.cpp
ARG SPACY_MODEL=https://github.com/explosion/spacy-models/releases/download/sv_core_news_sm-3.8.0/sv_core_news_sm-3.8.0-py3-none-any.whl

# ------------------------------------------------------------ pip images

FROM python:3.12-slim AS python
ENV HF_HOME=/cache/huggingface PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /work
COPY pyproject.toml README.md ./
COPY src src
ENTRYPOINT ["understudy"]

FROM python AS cpu
ARG SPACY_MODEL
COPY docker/constraints.txt docker/
# CPU torch, and the dataset stack pinned to a set that passes the tests
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --extra-index-url https://download.pytorch.org/whl/cpu -c docker/constraints.txt \
      -e ".[dataset,bot]" protobuf "$SPACY_MODEL"
COPY tests tests

FROM python AS train
# gcc: Triton compiles small launchers when Unsloth trains
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev \
 && rm -rf /var/lib/apt/lists/*
# torch from PyPI is a CUDA build on Linux; one install lets Unsloth pick its version
RUN --mount=type=cache,target=/root/.cache/pip pip install -e ".[train]"

# ------------------------------------------------------ llama.cpp images

FROM scratch AS code
COPY src/understudy /understudy

FROM ${LLAMA_CPP}:full AS convert
COPY --from=code /understudy /opt/understudy/understudy
ENV PYTHONPATH=/opt/understudy HF_HOME=/cache/huggingface
WORKDIR /work
ENTRYPOINT ["python3", "-m", "understudy"]

FROM ${LLAMA_CPP}:server AS serve
RUN apt-get update && apt-get install -y --no-install-recommends python3 \
 && rm -rf /var/lib/apt/lists/*
COPY --from=code /understudy /opt/understudy/understudy
ENV PYTHONPATH=/opt/understudy HF_HOME=/cache/huggingface PATH=/app:$PATH
WORKDIR /work
ENTRYPOINT ["python3", "-m", "understudy"]

FROM ${LLAMA_CPP}:server-cuda AS serve-cuda
RUN apt-get update && apt-get install -y --no-install-recommends python3 \
 && rm -rf /var/lib/apt/lists/*
COPY --from=code /understudy /opt/understudy/understudy
ENV PYTHONPATH=/opt/understudy HF_HOME=/cache/huggingface PATH=/app:$PATH
WORKDIR /work
ENTRYPOINT ["python3", "-m", "understudy"]
