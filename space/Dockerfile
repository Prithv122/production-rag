# Container image for the Gradio demo, targeted at Google Cloud Run.
#
# This is a *deployment* artefact. It installs the same `production_rag` package
# the evaluation measured and runs the same `space/app.py`; nothing about the
# retrieval or generation path changes because it is in a container.
#
# Build from the repository root, not from this directory -- the image needs
# `pyproject.toml` and `src/`:
#
#     docker build -f space/Dockerfile -t production-rag .
#
# Three decisions are worth stating.
#
# **CPU-only torch.** `pip install sentence-transformers` on Linux pulls the CUDA
# build of torch by default: roughly 2.5 GB of NVIDIA runtime for a machine that
# has no GPU. The `--extra-index-url .../whl/cpu` below is what keeps the image
# small enough that a cold start is tolerable.
#
# **The index and the encoder are baked at build time, not fetched at boot.**
# The index still comes from the pinned Hugging Face dataset -- that is the
# source of truth -- but it is downloaded during `docker build` rather than on
# every cold start. Measured locally, a from-scratch load that downloads the
# 61 MB index took 256 s; Cloud Run's default startup probe gives a container
# 240 s to accept traffic. Fetching at boot would therefore be a coin flip on
# whether the revision ever goes healthy. Baking makes startup model-load only.
# The cost is that refreshing the index means rebuilding the image, which is the
# right trade for an artefact whose whole point is that it does not drift.
#
# **No secret is baked in.** `OPENROUTER_API_KEY` is deliberately absent from
# this file and from the repository. Set it as a Cloud Run secret; without it
# the app serves retrieval and says plainly that generation is off.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf \
    HF_HUB_DISABLE_TELEMETRY=1 \
    INDEX_DIR=/opt/index \
    CACHE_DIR=/tmp/production-rag \
    PORT=8080

WORKDIR /app

# The package itself, plus the two things only the demo needs. Versions are
# pinned to what was actually verified locally rather than to a range.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        ".[embed]" \
        "gradio==6.26.0" \
        "huggingface-hub==1.30.0"

# Bake the pinned index (61 MB) and the encoder (~130 MB) into the image. Both
# are content-addressed upstream, so this pins the demo to exactly the artefacts
# the published numbers came from.
ARG INDEX_REPO=Prithv122/production-rag-index
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='${INDEX_REPO}', repo_type='dataset', local_dir='/opt/index')"
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-small-en-v1.5')"

COPY space/app.py ./app.py

# Cloud Run sets $PORT and routes to it; 8080 is its default.
EXPOSE 8080

# Run as a non-root user. CACHE_DIR lives under /tmp, which is writable.
RUN useradd --create-home --uid 1000 app && chown -R app:app /app /opt/hf
USER app

CMD ["python", "app.py"]
