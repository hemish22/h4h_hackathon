#!/usr/bin/env bash
# Bootstrap the GPU machine for training. Run ON the GPU box after cloning.
#   git clone https://github.com/hemish22/h4h_hackathon.git && cd h4h_hackathon
#   bash scripts/gpu_bootstrap.sh
# Then scp the dataset from the laptop (see scripts/push_dataset.sh) and train.
set -euo pipefail

echo "== python =="
python3 --version

echo "== venv =="
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip -q
python -m pip install -r requirements.txt

echo "== torch / CUDA =="
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

echo "== smoke test (no dataset needed) =="
python -m tests.test_smoke

echo
echo "Bootstrap done. Next:"
echo "  1) copy dataset/ from the laptop (run scripts/push_dataset.sh ON THE LAPTOP)"
echo "  2) python -m src.audit          # verify counts on this box"
echo "  3) python -m src.train --config configs/final.yaml"
