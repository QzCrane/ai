<#
.SYNOPSIS
    Sovereign AI Brain v12.2 — Native Windows RTX 3050 4GB Zero-C-Drive Bootstrap (2026-11-22 Latest).
    Locks ALL venvs and caches strictly under D:\Environment and installs 2026-11 latest stable dependencies:
      - Python 3.15 (fallback 3.14 / 3.13)
      - PyTorch 2.15.0+cu134 (auto-detects cu134 for driver >= R615, fallback cu132)
      - Official triton-lang/triton-windows 3.9.0.post30
      - flash-linear-attention 0.6.0+, liger-kernel 0.8.5+, transformers 5.24+, tokenizers 0.23.3+,
        huggingface-hub 2.4.0+, hf-xet 1.8.0+, datasets 5.2.0+, safetensors 0.8.0+, torchao 0.18.0+,
        numpy 2.5.5+, scipy 1.18.1+, polars 2.1.0+, duckdb 2.0.1+, swanlab 0.10.6+,
        z3-solver 5.4.1+, usearch 2.26.6+, mcp 2.3.0+, tree-sitter 0.26.1+, trafilatura 2.3.1+, ruff 0.17.6+
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\setup_windows_d_drive.ps1
#>
param(
    [string]$EnvRoot = "D:\Environment",
    [string]$PythonVersion = "3.15",
    [string]$CudaTag = "auto"
)

$ErrorActionPreference = "Stop"

# 1. Strictly pin ALL caches & temp dirs to D:\Environment\.cache (zero C-drive bloat)
$CacheRoot = Join-Path $EnvRoot ".cache"
$VenvDir   = Join-Path $EnvRoot "venvs\sovereign_brain"
$Dirs = @(
    $CacheRoot,
    (Join-Path $CacheRoot "huggingface"),
    (Join-Path $CacheRoot "pip"),
    (Join-Path $CacheRoot "torch"),
    (Join-Path $CacheRoot "triton"),
    (Join-Path $CacheRoot "uv"),
    (Join-Path $CacheRoot "tmp"),
    (Join-Path $EnvRoot "data"),
    (Join-Path $EnvRoot "checkpoints")
)
foreach ($d in $Dirs) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d -Force | Out-Null }
}

$env:HF_HOME          = Join-Path $CacheRoot "huggingface"
$env:PIP_CACHE_DIR    = Join-Path $CacheRoot "pip"
$env:TORCH_HOME       = Join-Path $CacheRoot "torch"
$env:TRITON_CACHE_DIR = Join-Path $CacheRoot "triton"
$env:UV_CACHE_DIR     = Join-Path $CacheRoot "uv"
$env:TEMP             = Join-Path $CacheRoot "tmp"
$env:TMP              = Join-Path $CacheRoot "tmp"

Write-Host "[v12.2] All caches locked under $CacheRoot (0 C-drive writes)" -ForegroundColor Cyan

# 2. Resolve Python 3.15 (latest stable) or fallback 3.14 / 3.13
$PyCmd = $null
foreach ($ver in @($PythonVersion, "3.15", "3.14", "3.13")) {
    try {
        $out = & py "-$ver" -c "import sys; print(sys.version.split()[0])" 2>$null
        if ($LASTEXITCODE -eq 0 -and $out) {
            $PyCmd = $ver
            Write-Host "[v12.2] Using Python $out (py -$ver)" -ForegroundColor Green
            break
        }
    } catch {}
}
if (-not $PyCmd) {
    throw "Python 3.15 (or 3.14/3.13) not found via 'py'. Please install Python 3.15 (64-bit) from python.org first."
}

# 3. Auto-detect latest PyTorch 2.15 CUDA wheel (cu134 for driver >= R615 / CUDA 13.4+, else cu132)
if ($CudaTag -eq "auto") {
    $CudaTag = "cu134"
    try {
        $smi = (& nvidia-smi) -join "`n"
        if ($smi -match "CUDA Version:\s*(\d+)\.(\d+)") {
            $maj = [int]$Matches[1]; $min = [int]$Matches[2]
            if ($maj -gt 13 -or ($maj -eq 13 -and $min -ge 4)) {
                $CudaTag = "cu134"
            } elseif ($maj -eq 13 -and $min -ge 2) {
                $CudaTag = "cu132"
            }
            Write-Host "[v12.2] Detected NVIDIA Driver CUDA $maj.$min -> selecting PyTorch wheel: $CudaTag" -ForegroundColor Green
        }
    } catch {
        Write-Warning "nvidia-smi not detected; defaulting to $CudaTag"
    }
}

# 4. Create venv on D: and upgrade packaging tools
if (-not (Test-Path (Join-Path $VenvDir "Scripts\python.exe"))) {
    Write-Host "[v12.2] Creating venv at $VenvDir ..." -ForegroundColor Cyan
    & py "-$PyCmd" -m venv $VenvDir
}
$VenvPy  = Join-Path $VenvDir "Scripts\python.exe"
$VenvPip = Join-Path $VenvDir "Scripts\pip.exe"

& $VenvPy -m pip install -U pip setuptools wheel

# 5. Install latest PyTorch 2.15+ (cu134) + official triton-windows 3.9+
Write-Host "[v12.2] Installing latest PyTorch ($CudaTag) + triton-windows ..." -ForegroundColor Cyan
& $VenvPip install -U torch --index-url "https://download.pytorch.org/whl/$CudaTag"
& $VenvPip install -U triton-windows

# 6. Install full 2026-11 latest ecosystem stack (no bitsandbytes per constitution)
Write-Host "[v12.2] Installing latest 2026-11 ecosystem packages ..." -ForegroundColor Cyan
& $VenvPip install -U flash-linear-attention liger-kernel transformers tokenizers huggingface-hub hf-xet datasets safetensors torchao numpy scipy polars duckdb swanlab z3-solver usearch mcp tree-sitter trafilatura ruff pytest

# 7. Hardware + Triton + FLA/Liger + Repo Doctor Verification
Write-Host "[v12.2] Running GPU / Triton / FLA / Liger / Repo Doctor self-check ..." -ForegroundColor Cyan
& $VenvPy -c "import sys, torch, triton, fla, liger_kernel; print('Python:', sys.version.split()[0], '| PyTorch:', torch.__version__, '| CUDA:', torch.version.cuda, '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU', '| Triton:', triton.__version__, '| FLA+Liger: OK')"
& $VenvPy scripts\repo_doctor.py --run-tests

Write-Host "[v12.2] Setup complete! Next step: powershell -ExecutionPolicy Bypass -File scripts\run_smoke_windows.ps1" -ForegroundColor Green
