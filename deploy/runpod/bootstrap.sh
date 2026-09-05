#!/usr/bin/env bash
# Bring the pod from "code extracted" to "server serving". Ships inside the
# tarball, runs after extraction, and is idempotent: on a pod restart every
# step that is already done is skipped, so a warm pod is serving in seconds
# rather than re-downloading 7 GB.
#
# Everything the boot loop redirects into /workspace/boot.log, which the
# receiver publishes on :8788/log. Nothing here prints a secret.
set -euo pipefail

WORKSPACE="${WORKSPACE_DIR:-/workspace}"
APP="$WORKSPACE/app/apps/brushjam"
MODELS="$WORKSPACE/models"
CHECKPOINT="${INPROC_CHECKPOINT:-$MODELS/checkpoints/sdxl-checkpoint.safetensors}"
CIVITAI_VERSION="${CIVITAI_VERSION:-2167369}"
MIN_CHECKPOINT_BYTES=6000000000

# The receiver publishes this file as `phase` in /status: one line, read with
# a strip(). deploy/runpod/test_deploy.py checks the two agree, because a
# receiver written by an older dockerStartCmd outlives every upload.
phase() { echo "$1" > "$WORKSPACE/phase"; echo "[bootstrap] === $1 ==="; }
have()  { command -v "$1" >/dev/null 2>&1; }

# Sourced by test_deploy.py to exercise phase() against the real receiver.
if [ -n "${BOOTSTRAP_FUNCTIONS_ONLY:-}" ]; then
  return 0 2>/dev/null || exit 0
fi

# The receiver SIGTERMs this pid to restart the server after a new upload.
echo $$ > "$WORKSPACE/server.pid"

phase "uv"
export PATH="$HOME/.local/bin:$PATH"
if ! have uv; then
  echo "[bootstrap] installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

phase "deps"
cd "$APP"
# The venv lives beside the code, not inside it. An upload replaces
# /workspace/app wholesale, and with uv's default .venv (inside the project,
# and so inside the swap) every upload paid for a full torch reinstall -
# several minutes for a change that should only restart the server. Here the
# environment survives the swap and `uv sync` is a no-op unless uv.lock moved.
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$WORKSPACE/venv}"
# uv fetches its own Python 3.10; uv.lock is committed, so this is reproducible.
uv sync --extra inproc

phase "checkpoint"
mkdir -p "$(dirname "$CHECKPOINT")" "$MODELS/loras" "${HF_HOME:-$WORKSPACE/hf}"
download_checkpoint() {
  local tmp="$CHECKPOINT.part"
  rm -f "$tmp"
  curl -fL --retry 3 --retry-delay 5 \
    -H "Authorization: Bearer ${CIVITAI_TOKEN:-}" \
    -o "$tmp" \
    "https://civitai.com/api/download/models/${CIVITAI_VERSION}?type=Model&format=SafeTensor"
  local size
  size=$(stat -c %s "$tmp")
  if [ "$size" -lt "$MIN_CHECKPOINT_BYTES" ]; then
    # Civitai answers an auth failure with a small HTML page and HTTP 200.
    echo "[bootstrap] checkpoint is only $size bytes; that is not the model"
    rm -f "$tmp"
    return 1
  fi
  mv "$tmp" "$CHECKPOINT"
  echo "[bootstrap] checkpoint downloaded: $size bytes"
}

if [ -f "$CHECKPOINT" ] && [ "$(stat -c %s "$CHECKPOINT")" -ge "$MIN_CHECKPOINT_BYTES" ]; then
  echo "[bootstrap] checkpoint already present ($(stat -c %s "$CHECKPOINT") bytes)"
else
  download_checkpoint || {
    echo "[bootstrap] retrying the checkpoint download once"
    sleep 5
    download_checkpoint
  }
fi

phase "server"
# The LoRA and the fp16-fix VAE are fetched by the pipeline itself through
# hf_hub_download, into HF_HOME on the persistent volume.
export HOST=0.0.0.0
export PORT=8787
export AI_BACKEND=inproc
export INPROC_CHECKPOINT="$CHECKPOINT"
export INPROC_LORA_DIR="${INPROC_LORA_DIR:-$MODELS/loras}"
export HF_HOME="${HF_HOME:-$WORKSPACE/hf}"
echo "[bootstrap] starting brushjam on :8787"
exec uv run brushjam
