#!/usr/bin/env bash
set -euo pipefail
REPO_URL="https://github.com/QzCrane/ai.git"
[ ! -d .git ] && git init -b main
git remote remove origin 2>/dev/null || true
git remote add origin "$REPO_URL"
git add .
git commit -m "feat: release Sovereign AI Brain (Route B v1.0) and Deterministic Agent Harness"
git push -u origin main
