#!/usr/bin/env bash
set -euo pipefail

REMOTE_URL="${REMOTE_URL:-https://github.com/limengran98/FORGE.git}"
BRANCH="${BRANCH:-main}"
COMMIT_MSG="${1:-update FORGE code}"

cd "$(dirname "$0")"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if [ -e ".git" ]; then
    if [ -d ".git" ] && [ -z "$(find .git -mindepth 1 -print -quit)" ]; then
      mv ".git" ".git.empty.$(date +%Y%m%d_%H%M%S)"
    else
      echo "[FORGE] Found an invalid non-empty .git path; refusing to overwrite it." >&2
      exit 1
    fi
  fi
  git init
fi

if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$REMOTE_URL"
else
  git remote add origin "$REMOTE_URL"
fi

if git rev-parse --verify HEAD >/dev/null 2>&1; then
  git branch -M "$BRANCH"
else
  git symbolic-ref HEAD "refs/heads/$BRANCH"
fi

CODE_PATHS=(
  ".gitignore"
  "README.md"
  "requirements.txt"
  "push_code_only.sh"
  "forge"
  "configs/forge_experiment.yaml"
  "configs/forge_llm.example.yaml"
  "configs/harness"
  "prompts"
  "skills"
  "tests"
  "workspace/initial_model.py"
)

git add -A -- "${CODE_PATHS[@]}"

if git diff --cached --quiet; then
  echo "[FORGE] No code changes to commit."
else
  git commit -m "$COMMIT_MSG"
fi

git push -u origin "$BRANCH"
