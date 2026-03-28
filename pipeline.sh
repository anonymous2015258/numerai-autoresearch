#!/usr/bin/env bash
# Numerai automated improvement pipeline
# Runs Claude Code in a loop, each iteration proposes + evaluates one experiment
#
# Usage:
#   ./pipeline.sh              # run indefinitely
#   ./pipeline.sh 20           # run 20 iterations then stop
#   ./pipeline.sh 0 --dry-run  # print the prompt without running

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAX_ITERATIONS="${1:-0}"   # 0 = unlimited
DRY_RUN="${2:-}"
LOG_FILE="$SCRIPT_DIR/pipeline.log"
ITERATION=0

# ── Resolve claude binary ─────────────────────────────────────────────────────

# 1. Prefer claude on PATH
# 2. Fall back to VS Code extension binary (latest version)
if command -v claude &>/dev/null; then
    CLAUDE_BIN="claude"
else
    CLAUDE_BIN=$(
        find "$HOME/.vscode/extensions" -path "*/native-binary/claude" -type f 2>/dev/null \
        | sort -V | tail -1
    )
    if [[ -z "$CLAUDE_BIN" ]]; then
        echo "ERROR: 'claude' CLI not found. Add it to PATH or install the Claude Code VS Code extension."
        exit 1
    fi
    echo "Using claude at: $CLAUDE_BIN"
fi

# ── Preflight checks ──────────────────────────────────────────────────────────

if ! git -C "$SCRIPT_DIR" rev-parse --git-dir &>/dev/null; then
    echo "ERROR: Not a git repository. Run 'git init && git add . && git commit -m init' first."
    exit 1
fi

if [[ ! -f "$SCRIPT_DIR/loop_prompt.md" ]]; then
    echo "ERROR: loop_prompt.md not found."
    exit 1
fi

if [[ ! -f "$SCRIPT_DIR/results.tsv" ]]; then
    echo "ERROR: results.tsv not found."
    exit 1
fi

# ── Dry run ───────────────────────────────────────────────────────────────────

if [[ "$DRY_RUN" == "--dry-run" ]]; then
    echo "=== Prompt that would be sent to Claude ==="
    cat "$SCRIPT_DIR/loop_prompt.md"
    exit 0
fi

# ── Main loop ─────────────────────────────────────────────────────────────────

cleanup() {
    echo ""
    echo "Pipeline stopped after $ITERATION iteration(s). See pipeline.log for full output."
}
trap cleanup EXIT INT TERM

echo "Starting pipeline. Logs → $LOG_FILE"
echo "Press Ctrl+C to stop."
echo ""

while true; do
    ITERATION=$((ITERATION + 1))
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

    echo "━━━ Iteration $ITERATION  [$TIMESTAMP] ━━━"
    echo "━━━ Iteration $ITERATION  [$TIMESTAMP] ━━━" >> "$LOG_FILE"

    # Run one Claude Code iteration (non-interactive)
    "$CLAUDE_BIN" \
        --allowedTools "Read,Edit,Write,Bash,Glob,Grep" \
        --model claude-sonnet-4-6 \
        -p "$(cat "$SCRIPT_DIR/loop_prompt.md")" \
        2>&1 | tee -a "$LOG_FILE"

    echo "" | tee -a "$LOG_FILE"

    # Stop if we've hit the requested number of iterations
    if [[ "$MAX_ITERATIONS" -gt 0 && "$ITERATION" -ge "$MAX_ITERATIONS" ]]; then
        echo "Reached $MAX_ITERATIONS iteration(s). Stopping."
        break
    fi

    # Short pause between iterations so Claude doesn't hammer resources
    sleep 5
done
