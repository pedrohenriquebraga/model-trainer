#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

SITE_PACKAGES="$("$PWD/.venv/bin/python" -c \
    'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
CUDA_LIBS="$(find "$SITE_PACKAGES/nvidia" \
    -mindepth 2 -maxdepth 2 -type d -name lib -printf '%p:' 2>/dev/null || true)"
CUDA_LIBS="${CUDA_LIBS%:}"
export LD_LIBRARY_PATH="${CUDA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

.venv/bin/python build.py