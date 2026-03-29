# Numerai AutoResearch Pipeline

An automated ML improvement pipeline for the [Numerai](https://numer.ai) tournament,
inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

A Claude Code agent runs in a loop: reads the current model, proposes one focused
change, trains, evaluates, keeps improvements, reverts failures — autonomously,
overnight, with no human writing model code.

**Results after 21 iterations: CORR Sharpe 0.84 → 1.56 (+85.5%), zero human-written model code.**

---

## How It Works

```
loop:
  1. Claude reads program.md + results.tsv + train.py
  2. Proposes ONE change not yet tried
  3. Edits train.py
  4. Runs python3.10 train.py  →  validation_predictions.parquet
  5. Runs python3.10 evaluate.py  →  metrics.json
  6. If improved: git commit + log to results.tsv
     If worse:    git restore train.py + log failure
  7. Repeat
```

Human role: update `program.md` with new research directions between sessions.

---

## Prerequisites

- Python 3.10 with packages: `numerapi pandas pyarrow lightgbm scikit-learn scipy numerai-tools`
- [Claude Code](https://claude.ai/code) installed (VS Code extension or CLI)
- Git

```bash
pip3 install numerapi pandas pyarrow lightgbm scikit-learn scipy "cloudpickle==3.1.1" numerai-tools
```

---

## Setup

```bash
git clone https://github.com/anonymous2015258/numerai-autoresearch
cd numerai-autoresearch

# Initialise git (already done if cloned)
git log --oneline | head -5
```

---

## Running

### 1. Establish a baseline first

Run the starter model once to download data and populate `results.tsv`:

```bash
python3.10 train.py
python3.10 evaluate.py
```

Then manually add a baseline row to `results.tsv`:
```
<git_hash>  <corr_mean>  <corr_sharpe>  <mmc_sharpe>  <max_drawdown>  success  baseline lgbm small features
```

### 2. Run the pipeline

```bash
# Run 10 iterations (recommended for a first session)
./pipeline.sh 10

# Run indefinitely (Ctrl+C to stop)
./pipeline.sh

# Preview the prompt without running
./pipeline.sh 0 --dry-run
```

Logs are written to `pipeline.log`. Monitor live:
```bash
tail -f pipeline.log
```

Check results anytime:
```bash
cat results.tsv
```

### 3. Steer research between sessions

Edit `program.md` to:
- Mark explored ideas as `[x]` so the agent doesn't retry them
- Add new directions to explore
- Update `BATCH_BASELINE:` when rolling back to a simpler model
- Adjust runtime constraints (ensemble size, downsampling)

---

## File Structure

```
.
├── train.py          ← The only file the agent edits
│                       Contains: feature set, model config, training loop
│                       Output:   validation_predictions.parquet
│                                 live_predictions.parquet
│
├── evaluate.py       ← Fixed evaluation harness (never edited)
│                       Input:  validation_predictions.parquet
│                       Output: metrics.json (CORR/MMC mean, std, sharpe, drawdown)
│
├── pipeline.sh       ← Shell loop that calls `claude -p` repeatedly
│                       Usage: ./pipeline.sh [N_iterations] [--dry-run]
│
├── loop_prompt.md    ← Per-iteration instructions given to the agent
│                       Defines the 10-step experiment procedure
│
├── program.md        ← Human-written research directions
│                       Updated by human between sessions
│
└── results.tsv       ← Experiment log
                        Columns: commit_hash, corr_mean, corr_sharpe,
                                 mmc_sharpe, max_drawdown, status, description
```

---

## Key Design Decisions

**`train.py` is the only editable file.**
The agent can change anything inside it — model type, features, targets,
hyperparameters, post-processing — as long as it outputs
`validation_predictions.parquet` with columns `[era, prediction, target]`.

**Git is the undo mechanism.**
Every successful experiment is a commit. Failed experiments are reverted with
`git restore train.py`. Full reproducibility: `git checkout <hash>` gives
you any experiment back.

**`program.md` is human strategy; `loop_prompt.md` is agent tactics.**
`program.md` tells the agent *what* to explore and *what to avoid*.
`loop_prompt.md` tells the agent *how* to run one iteration.
Only update `program.md` between sessions.

**`BATCH_BASELINE:` override.**
When rolling back `train.py` to a simpler model, add this to `program.md`:
```
BATCH_BASELINE: 1.4786
```
Without it, the agent compares against the all-time best in `results.tsv`,
which may be from a model you've already discarded.

---

## Results (21 iterations)

| Iteration | Change | CORR Sharpe | Status |
|---|---|---|---|
| 1 | Baseline (small features, LightGBM) | 0.8415 | ✅ |
| 3 | **small → medium features (42→780)** | **1.3808** | ✅ |
| 5 | n_estimators 5000, lr 0.005 | 1.4001 | ✅ |
| 7 | subsample=0.8 | 1.4056 | ✅ |
| 8 | Every 2nd era (2× training data) | 1.4313 | ✅ |
| 10 | max_depth=6, num_leaves=63 | 1.4544 | ✅ |
| 12 | max_depth=7, num_leaves=127 | 1.4786 | ✅ |
| 13 | 3-target ensemble | 1.5504 | ✅ |
| 16 | **2-target ensemble (target+cyrusd_20)** | **1.5527** | ✅ |
| 19 | **colsample_bytree=0.05** | **1.5615** | ✅ best |

See `pipeline.log` for the full run history and `results.tsv` for raw metrics.

---

## Common Issues

| Problem | Cause | Fix |
|---|---|---|
| `claude: command not found` | Claude Code not on PATH | Script auto-discovers from `~/.vscode/extensions` |
| `ModuleNotFoundError: numerapi` | Wrong Python version | Use `python3.10` explicitly |
| Iteration stuck >60 min | Agent ran training in background (`tail -f` deadlock) | `loop_prompt.md` says foreground-only; watchdog kills at 60 min |
| All experiments failing | `results.tsv` baseline higher than current `train.py` | Add `BATCH_BASELINE: X.XXXX` to `program.md` |
| Wrong target name crash | Approximate name in `program.md` | Use exact column names from `v5.2/features.json` |

---

## Numerai Tournament

- Sign up: [numer.ai](https://numer.ai)
- Data docs: [docs.numer.ai](https://docs.numer.ai)
- The tournament runs weekly; submit live predictions Tuesday–Saturday
