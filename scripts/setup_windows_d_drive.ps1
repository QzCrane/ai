<#
.SYNOPSIS
    Sovereign AI Brain v12.2 — Native Windows RTX 3050 4GB Zero-C-Drive Bootstrap (2026-11-22 Latest).
    Locks ALL venvs and caches strictly under D:\Environment and installs 2026-11 latest stable dependencies:
      - Python 3.14 (3.15 is NOT usable: TCC inside triton-windows cannot compile against
        CPython 3.15 headers -> "no supported thread-local variable storage classifier")
      - PyTorch 2.15.0+cu134 (auto-detects cu134 for driver >= R615, fallback cu132)
      - Official triton-lang/triton-windows 3.9.0.post30
      - flash-linear-attention from source (PyPI wheel lacks fla/ops), liger-kernel --no-deps,
        transformers 5.24+, tokenizers 0.23.3+, huggingface-hub 2.4.0+, hf-xet 1.8.0+,
        datasets 5.2.0+, safetensors 0.8.0+, torchao 0.18.0+,
        numpy 2.5.5+, scipy 1.18.1+, polars 2.1.0+, duckdb 2.0.1+, swanlab 0.10.6+,
        z3-solver 5.4.1+, usearch 2.26.6+, mcp 2.3.0+, tree-sitter 0.26.1+, trafilatura 2.3.1+, ruff 0.17.6+
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\setup_windows_d_drive.ps1
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\setup_windows_d_drive.ps1 -PythonExe "D:\Environment\toolchains\python\3.14.7\python.exe"
#>
param(
    [string]$EnvRoot = "D:\Environment",
    [string]$PythonVersion = "3.14",
    [string]$PythonExe = "",
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

# 2. Resolve a WORKING Python interpreter: -PythonExe > py launcher > uv-managed 3.14
#    NOTE (measured 2026-11-22): CPython 3.15 cannot run triton-windows 3.9.0.post30.
#    triton-windows ships TCC to JIT-compile its CUDA driver shim, and CPython 3.15's
#    pyport.h hard-errors with "no supported thread-local variable storage classifier"
#    for toolchains that are neither MSVC (__declspec(thread)) nor GCC (__thread).
#    Therefore 3.14 is the newest interpreter on which fla / Liger Triton kernels work.
$PyExe = $null
if ($PythonExe) {
    if (-not (Test-Path $PythonExe)) { throw "-PythonExe not found: $PythonExe" }
    $PyExe = $PythonExe
    Write-Host "[v12.2] Using explicit interpreter $PyExe" -ForegroundColor Green
}
if (-not $PyExe) {
    foreach ($ver in @($PythonVersion, "3.14", "3.13")) {
        try {
            $out = & py "-$ver" -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $out) {
                $PyExe = ($out | Select-Object -Last 1).Trim()
                Write-Host "[v12.2] Using Python $ver via py launcher: $PyExe" -ForegroundColor Green
                break
            }
        } catch {}
    }
}
if (-not $PyExe -and (Test-Path (Join-Path $EnvRoot "toolchains\python\3.14.7\python.exe"))) {
    $PyExe = Join-Path $EnvRoot "toolchains\python\3.14.7\python.exe"
    Write-Host "[v12.2] Using D-drive interpreter: $PyExe" -ForegroundColor Green
}
if (-not $PyExe) {
    # uv-managed fallback (no py launcher, no system Python on D:)
    $Uv = $null
    foreach ($cand in @((Join-Path $EnvRoot "toolchains\uv-win\uv.exe"), "uv")) {
        try { if ($cand -eq "uv" -or (Test-Path $cand)) { $Uv = $cand; break } } catch {}
    }
    if ($Uv) {
        $env:UV_CACHE_DIR = Join-Path $CacheRoot "uv\win-amd64"
        $env:UV_PYTHON_INSTALL_DIR = Join-Path $EnvRoot "toolchains\uv-python"
        & $Uv python install $PythonVersion 2>&1 | Out-Host
        $PyExe = (& $Uv python find $PythonVersion 2>$null | Select-Object -Last 1).Trim()
        Write-Host "[v12.2] Using uv-managed Python: $PyExe" -ForegroundColor Green
    }
}
if (-not $PyExe) {
    throw "No usable Python found. Install Python $PythonVersion (64-bit) or pass -PythonExe <path>."
}
$PyCmd = $PyExe

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
    & $PyExe -m venv $VenvDir
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
& $VenvPip install -U transformers tokenizers huggingface-hub hf-xet datasets safetensors torchao numpy scipy polars duckdb swanlab z3-solver usearch mcp tree-sitter trafilatura ruff pytest einops

# 6b. fla / Liger on Windows: PyPI wheels are NOT sufficient.
#     - `flash-linear-attention` 0.6.0 on PyPI ships only fla/{layers,models}; `fla.ops` is
#       missing (0.4 MB wheel), so any kernel dispatch silently degrades. Install from source.
#     - `liger-kernel` hard-depends on the Linux-only `triton` distribution; on Windows the
#       `triton` module is provided by `triton-windows`, so resolve with --no-deps.
Write-Host "[v12.2] Installing fla (source) + liger-kernel (no-deps) ..." -ForegroundColor Cyan
& $VenvPip install -U --no-deps liger-kernel
& $VenvPip install -U --no-deps --force-reinstall "flash-linear-attention @ git+https://github.com/fla-org/flash-linear-attention"

# 7. Hardware + Triton + FLA/Liger + Repo Doctor Verification
Write-Host "[v12.2] Running GPU / Triton / FLA / Liger / Repo Doctor self-check ..." -ForegroundColor Cyan
& $VenvPy -c "import sys, torch, triton; print('Python:', sys.version.split()[0], '| PyTorch:', torch.__version__, '| CUDA:', torch.version.cuda, '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU', '| Triton:', triton.__version__)"
& $VenvPy -c "import torch; from fla.ops.gated_delta_rule import chunk_gated_delta_rule; from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss; B,H,T,D=1,4,64,32; q,k,v=[torch.randn(B,H,T,D,device='cuda',dtype=torch.bfloat16) for _ in range(3)]; g=torch.randn(B,H,T,D,device='cuda',dtype=torch.bfloat16); be=torch.randn(B,H,T,device='cuda',dtype=torch.bfloat16); r=chunk_gated_delta_rule(q,k,v,g,be); W=torch.randn(256,64,device='cuda',dtype=torch.bfloat16,requires_grad=True); Hs=torch.randn(8,64,device='cuda',dtype=torch.bfloat16,requires_grad=True); Tg=torch.randint(0,256,(8,),device='cuda'); l=LigerFusedLinearCrossEntropyLoss()(W,Hs,Tg); print('FLA real kernel: OK | Liger real kernel: OK | loss=%.4f' % float(l))"
& $VenvPy scripts\repo_doctor.py --run-tests

Write-Host "[v12.2] Setup complete! Next step: powershell -ExecutionPolicy Bypass -File scripts\run_smoke_windows.ps1" -ForegroundColor Green
