# Part Assistant image. Data, models and the database live in volumes (see compose.yaml),
# so this image holds only Python, the locked packages and the source code.
#
# torch is the CUDA 12.8 build (uv.lock): it runs on CPU too, and on an NVIDIA GPU when
# the container gets one (compose.gpu.yaml). That build makes the image large (~8 GB).

FROM python:3.12-slim-bookworm

# uv from its official image, same version as CI.
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1

# Non-root user. The data/cache folders are created here, owned by that user, so the
# named volumes mounted on them start with the right owner.
RUN useradd --create-home --uid 1000 app \
 && mkdir -p /app/data/raw /app/data/processed /app/traces /app/runs /cache \
 && chown -R app:app /app /cache
WORKDIR /app
USER app

# 1) Packages only (the slow layer: ~7 GB). Rebuilt only when uv.lock / pyproject change.
RUN --mount=type=cache,target=/home/app/.cache/uv,uid=1000 \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --no-install-project

# 2) The project itself (fast layer): rebuilt when the code changes.
COPY --chown=app:app . .
RUN --mount=type=cache,target=/home/app/.cache/uv,uid=1000 \
    uv sync --locked --no-dev

ENV PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/cache/huggingface \
    XDG_CACHE_HOME=/cache

ENTRYPOINT ["/app/docker/entrypoint.sh"]
