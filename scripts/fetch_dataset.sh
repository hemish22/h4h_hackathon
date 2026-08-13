#!/usr/bin/env bash
# Fetch the raw dataset (not in git) from the GitHub release into ./dataset.
# Run from the repo root after cloning:  bash scripts/fetch_dataset.sh
set -euo pipefail

if [ -f dataset/mental_health_multimodal.csv ]; then
  echo "dataset/ already present — nothing to do."
  exit 0
fi

URL="https://github.com/hemish22/h4h_hackathon/releases/download/data-v1/dataset.tgz"
echo "downloading dataset (~476MB) ..."
if command -v wget >/dev/null 2>&1; then
  wget -q --show-progress -O dataset.tgz "$URL"
else
  curl -L -o dataset.tgz "$URL"
fi

echo "extracting ..."
tar --warning=no-unknown-keyword -xzf dataset.tgz 2>/dev/null || tar -xzf dataset.tgz
find dataset -name '._*' -delete 2>/dev/null || true
rm -f dataset.tgz
echo "done. verify:  python -m src.audit"
