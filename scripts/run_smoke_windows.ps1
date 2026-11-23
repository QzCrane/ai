<#
.SYNOPSIS
    Sovereign AI Brain v12.2 — One-Click 200MB Super-Density Smoke Corpus + RTX 3050 4GB Steady-State Benchmark.
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\run_smoke_windows.ps1
#>
param(
    [string]$EnvRoot = "D:\Environment",
    [int]$TargetMB = 200,
    [int]$BatchSize = 4,
    [int]$SeqLen = 512,
    [int]$WarmupSteps = 3,
    [int]$Steps = 20
)

$ErrorActionPreference = "Stop"

$CacheRoot = Join-Path $EnvRoot ".cache"
$VenvPy    = Join-Path $EnvRoot ".venvs\sovereign_brain\Scripts\python.exe"
$DataDir   = Join-Path $EnvRoot ".data\seed_v12_smoke"
$CkptPath  = Join-Path $EnvRoot ".checkpoints\sovereign_rtx3050_smoke.pt"

if (-not (Test-Path $VenvPy)) {
    throw "Venv not found at $VenvPy. Run scripts\setup_windows_d_drive.ps1 first."
}

$env:HF_HOME          = Join-Path $CacheRoot "huggingface"
$env:PIP_CACHE_DIR    = Join-Path $CacheRoot "pip"
$env:TORCH_HOME       = Join-Path $CacheRoot "torch"
$env:TRITON_CACHE_DIR = Join-Path $CacheRoot "triton"
$env:TEMP             = Join-Path $EnvRoot ".tmp"
$env:TMP              = Join-Path $EnvRoot ".tmp"

Write-Host "[1/3] Building ${TargetMB}MB super-density smoke slice (O(1) SimHash + 80% Tier-S real anchor + 20% Causal-IR/Abstention) ..." -ForegroundColor Cyan
& $VenvPy src\brain\prepare_seed_corpus.py --profile rtx3050_4gb_super_dense_v11 --super-density --use-ube --target-mb $TargetMB --output-dir $DataDir

Write-Host "[2/3] Verifying Zero-Init External Backbone Engram Adapter ..." -ForegroundColor Cyan
& $VenvPy src\brain\sovereign_brain_pytorch.py --verify-engram-adapter

Write-Host "[3/3] Running $Steps-step steady-state RTX 3050 4GB benchmark (batch=$BatchSize, seq=$SeqLen, warmup=$WarmupSteps) ..." -ForegroundColor Cyan
$BinFile = Join-Path $DataDir "seed_corpus.bin"
& $VenvPy src\brain\sovereign_brain_pytorch.py --config rtx3050_4gb_65m --data-bin $BinFile --batch-size $BatchSize --seq-len $SeqLen --warmup-steps $WarmupSteps --steps $Steps --save-path $CkptPath

Write-Host "[v12.2] Smoke benchmark finished cleanly! Checkpoint saved to $CkptPath" -ForegroundColor Green
