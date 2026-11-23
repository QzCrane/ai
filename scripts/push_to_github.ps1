$ErrorActionPreference = "Stop"
$RepoUrl = "https://github.com/QzCrane/ai.git"
if (-not (Test-Path ".git")) { git init -b main }
git remote remove origin 2>$null
git remote add origin $RepoUrl
git add .
git commit -m "feat: release Sovereign AI Brain (Route B v1.0) and Deterministic Agent Harness"
git push -u origin main
