# Pipeline Lessons — Numerai AutoResearch

Problems encountered and fixes made while building an autoresearch-style
automated ML improvement pipeline using Claude Code as the agent.

---

## Problem 1: `claude` CLI not on PATH

**What happened:**
`pipeline.sh` called `claude` directly. When run as a subprocess, the shell
had no PATH entry for it — Claude Code is bundled inside the VS Code extension,
not installed as a system binary.

**Error:**
```
ERROR: 'claude' CLI not found. Install Claude Code first.
```

**Fix applied to `pipeline.sh`:**
Auto-discover the binary from the VS Code extensions directory, sorted by
version to get the latest:
```bash
CLAUDE_BIN=$(
    find "$HOME/.vscode/extensions" -path "*/native-binary/claude" -type f \
    | sort -V | tail -1
)
```

---

## Problem 2: Two Python environments, wrong one used

**What happened:**
The machine had two Pythons: system Python 3.9 (`/usr/bin/python3`) with no
packages, and Python 3.10 (`/Library/Frameworks/...`) with all ML packages.
`loop_prompt.md` originally said `python train.py`, which hit the wrong one.

**Error:**
```
ModuleNotFoundError: No module named 'numerapi'
```

**Fix applied to `loop_prompt.md` and `train.py`:**
Explicitly use `python3.10`:
```bash
python3.10 train.py
python3.10 evaluate.py
```

---

## Problem 3: `tail -f` deadlock — iteration stuck for 1h47min

**What happened:**
The agent ran `python3.10 train.py` as a **background** Bash command and
monitored its output with `tail -f`. Training finished but `tail -f` never
exits on a completed file — it polls indefinitely. The agent waited forever.

**Symptom:**
No `python3.10` process visible, but the `claude` subprocess had a child
`/bin/zsh -c ... tail -f <task_output_file>` running with 0.0% CPU.
Iteration ran for 1 hour 47 minutes without completing.

**Fix applied to `loop_prompt.md`:**
```
IMPORTANT: run this as a foreground command — do NOT use background
execution or `run_in_background=true`.
```

---

## Problem 4: `timeout` command not available on macOS

**What happened:**
Added `timeout 3600 claude ...` to `pipeline.sh` as a safeguard.
`timeout` is a GNU coreutils command — not available on macOS BSD shell.

**Error:**
```
./pipeline.sh: line 81: timeout: command not found
[pipeline] iteration 1 timed out or errored — continuing
```
Both iterations completed instantly (errored out immediately), no training ran.

**Fix applied to `pipeline.sh`:**
Replaced with a macOS-native watchdog using a background `sleep` + `kill`:
```bash
"$CLAUDE_BIN" ... &
CLAUDE_PID=$!
( sleep 3600; kill "$CLAUDE_PID" 2>/dev/null ) &
WATCHDOG_PID=$!
wait "$CLAUDE_PID" || true
kill "$WATCHDOG_PID" 2>/dev/null || true
```

---

## Problem 5: Wrong target name caused crash

**What happened:**
`program.md` listed `target_cyrus` as a target to try. The agent used it
literally. The actual column name in v5.2 is `target_cyrusd_20`.

**Error:**
`KeyError: 'target_cyrus'` — train.py crashed, logged as `crashed` in results.tsv.

**Fix applied to `program.md`:**
Listed all targets with exact column names:
```
target_cyrusd_20, target_victor_20, target_jerome_20 ...
```

---

## Problem 6: 3-model ensemble blew up iteration time to 60+ min

**What happened:**
The agent discovered target ensembling (train one model per target, average
predictions) and got a big win: CORR Sharpe 1.4786 → 1.5504. But training
3 × LightGBM models with 5000 estimators on every-2nd-era data took **60–70
minutes per iteration** — 3× longer than expected. One iteration ran for
1h47min before being killed.

**Fix applied to `program.md`:**
```
- Ensembles (multiple targets or models): set DOWNSAMPLE_EVERY_N_ERAS = 4
- Single model experiments: DOWNSAMPLE_EVERY_N_ERAS = 2 is fine
- Max 2 targets in any ensemble
```

---

## Problem 7: Two pipelines running concurrently after restarts

**What happened:**
After killing and restarting the pipeline multiple times, old `pipeline.sh`
processes didn't always die cleanly. Two simultaneous pipelines competed to
edit `train.py`, commit, and write `results.tsv` — causing race conditions.

**Fix:**
Before each restart: `kill <pipeline_pid> <claude_pid>` and confirm with
`ps aux | grep pipeline.sh` before launching a fresh run.

---

## Problem 8: Baseline mismatch after rolling back train.py

**What happened:**
We rolled back `train.py` from the slow 3-target ensemble (CORR Sharpe 1.550)
back to the single-model baseline (1.4786) to keep iteration times fast.
But `results.tsv` still had 1.550357 as the best `success` row.
The agent compared every new experiment against 1.550 — all single-model
experiments "failed" even when they were actually improvements over the
single-model baseline. The pipeline made zero progress for multiple iterations.

**Fix applied to `loop_prompt.md`:**
```
Check program.md for a line starting with BATCH_BASELINE: — if present,
use that value as the reference score instead of scanning results.tsv.
```

**Fix applied to `program.md`:**
```
BATCH_BASELINE: 1.4786
```

---

## Results summary (15 iterations)

| Iteration | Description | CORR Sharpe | Status |
|---|---|---|---|
| 1 | baseline lgbm small features | 0.8415 | ✅ |
| 2 | rank predictions within era | 0.8415 | ❌ |
| 3 | **switch small → medium features** | **1.3808** | ✅ |
| 4 | colsample_bytree 0.1→0.3 | 1.3718 | ❌ |
| 5 | n_estimators 5000, lr 0.005 | 1.4001 | ✅ |
| 6 | target_cyrus (wrong name) | 0 | 💥 crashed |
| 7 | subsample=0.8, subsample_freq=1 | 1.4056 | ✅ |
| 8 | downsample every 2nd era | 1.4313 | ✅ |
| 9 | use all eras (no downsampling) | 1.3519 | ❌ |
| 10 | max_depth 6, num_leaves 63 | 1.4544 | ✅ |
| 11 | min_child_samples=200 | 1.4513 | ❌ |
| 12 | max_depth 7, num_leaves 127 | **1.4786** | ✅ |
| 13 | **3-target ensemble** | **1.5504** | ✅ (slow) |
| 14 | per-era mean-centering | 1.4786 | ❌ vs ensemble |
| 15 | XGBoost + LightGBM ensemble | 1.4786 | ❌ vs ensemble |

**Best single model: 1.4786** (+75.6% over baseline 0.8415)
**Best overall: 1.5504** (3-target ensemble, but 60+ min/iter)
