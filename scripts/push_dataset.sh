#!/usr/bin/env bash
# Copy the raw dataset/ from THIS laptop to the GPU box over SSH.
# dataset/ is gitignored (~200MB) so it must be transferred out-of-band.
#
# Usage:  bash scripts/push_dataset.sh user@host[:port] [remote_repo_path]
#   e.g.  bash scripts/push_dataset.sh alex@192.168.1.42 ~/h4h_hackathon
set -euo pipefail

TARGET="${1:?usage: push_dataset.sh user@host[:port] [remote_repo_path]}"
REMOTE_PATH="${2:-h4h_hackathon}"

# split optional :port
PORT=22
if [[ "$TARGET" == *:* ]]; then
  PORT="${TARGET##*:}"
  TARGET="${TARGET%%:*}"
fi

echo "rsync dataset/ -> $TARGET:$REMOTE_PATH/dataset/ (port $PORT)"
rsync -avz --progress -e "ssh -p $PORT" \
  ./dataset/ "$TARGET:$REMOTE_PATH/dataset/"

echo "done. verify on the box:  python -m src.audit"
