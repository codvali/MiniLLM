#!/bin/bash
# pull.sh <hf_repo> <gguf_file> [dest_dir]
# ex: ./pull.sh Qwen/Qwen3-14B-GGUF Qwen3-14B-Q4_K_M.gguf
# downloads a GGUF straight from HuggingFace — no ollama pull needed.
# With auto_dir configured, the model is served automatically.
REPO="$1"; FILE="$2"
DEST="${3:-$(cd "$(dirname "$0")" && pwd)/models}"
[ -z "$FILE" ] && { echo "usage: pull.sh <repo> <file> [dir]"; exit 1; }
mkdir -p "$DEST"
URL="https://huggingface.co/$REPO/resolve/main/$FILE"
OUT="$DEST/$FILE"
echo "[pull] $URL -> $OUT"
curl -fL --progress-bar -o "$OUT.part" "$URL" \
  && mv "$OUT.part" "$OUT" && echo "[pull] done: $(du -h "$OUT" | cut -f1)"
