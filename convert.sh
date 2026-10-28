#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ -x "$PWD/.venv/bin/python" ]; then
    SITE_PACKAGES="$("$PWD/.venv/bin/python" -c \
        'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
else
    SITE_PACKAGES="$(python3 -c \
        'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
fi
CUDA_LIBS="$(find "$SITE_PACKAGES/nvidia" \
    -mindepth 2 -maxdepth 2 -type d -name lib -printf '%p:' 2>/dev/null || true)"
CUDA_LIBS="${CUDA_LIBS%:}"
export LD_LIBRARY_PATH="${CUDA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [ -x ".venv/bin/python" ]; then
    exec .venv/bin/python convert.py --quantization float16 --runtime builtin "$@"
fi

exec python3 convert.py --quantization float16 --runtime builtin "$@"