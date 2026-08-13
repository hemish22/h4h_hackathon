# Bootstrap the GPU machine (Windows / PowerShell) for training.
# Run ON the GPU box after cloning:
#   git clone https://github.com/hemish22/h4h_hackathon.git
#   cd h4h_hackathon
#   powershell -ExecutionPolicy Bypass -File scripts\gpu_bootstrap.ps1
$ErrorActionPreference = "Stop"

Write-Host "== python ==" -ForegroundColor Cyan
python --version

Write-Host "== venv ==" -ForegroundColor Cyan
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip -q
python -m pip install -r requirements.txt

Write-Host "== torch / CUDA ==" -ForegroundColor Cyan
python -c "import torch; print('torch', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

Write-Host "== smoke test (no dataset needed) ==" -ForegroundColor Cyan
python -m tests.test_smoke

Write-Host ""
Write-Host "Bootstrap done. Next:" -ForegroundColor Green
Write-Host "  1) receive dataset.tgz from the laptop, then:  tar -xzf dataset.tgz"
Write-Host "  2) python -m src.audit"
Write-Host "  3) python -m src.train --config configs/final.yaml"
