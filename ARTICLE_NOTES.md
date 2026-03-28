# X Article Notes — Numerai AutoResearch Pipeline

---

## 1. The Idea & Objective

Most data science competitions are won by humans grinding through experiments
manually: try a feature, retrain, check the score, repeat. It is slow, requires
constant attention, and is bounded by how many ideas one person can test in a day.

**The idea:** replace the human in that loop with an AI agent.

Inspired by Andrej Karpathy's `autoresearch` project — where a Claude agent
autonomously rewrites a GPT training script overnight and discovers architectural
improvements without human intervention — we asked: can the same approach work
for a real-world financial ML competition?

**The objective:**
- Build a pipeline that runs fully autonomously overnight
- Each iteration: the agent reads the current model, proposes ONE change,
  trains, evaluates, keeps improvements, reverts failures
- Measure how much CORR Sharpe (the competition's primary metric) improves
  without any human writing a single line of model code

The baseline (a simple LightGBM starter) had a CORR Sharpe of **0.84**.
The question was: how far can an AI agent push it?

---

## 2. What Is the Numerai Competition?

Numerai is a hedge fund that runs what it calls "the hardest data science
tournament in the world." Every week, data scientists around the world submit
stock market predictions. Numerai stakes real money on the aggregated meta-model,
and pays out based on how well individual predictions correlate with it.

### The Dataset

- **Fully obfuscated** — stock IDs, feature names, and target definitions are
  all anonymized. You cannot use financial intuition or domain knowledge.
  This forces pure ML.
- **~1,500 eras** (weeks) of historical data, each with ~500–2,000 stocks
- **2,748 features** available (P/E ratios, RSI, short interest, analyst
  ratings — all binned to integers 0–4)
- **41 target variables** measuring stock-specific returns over 20 and 60
  business days forward
- **Live predictions** submitted weekly on the current state of the market

### Feature Sets
| Set | Features | Use case |
|---|---|---|
| small | 42 | Fast prototyping |
| medium | 780 | Balanced performance |
| all | 2,748 | Maximum signal |

### What You're Predicting

The `target` is a measure of **stock-specific** future returns — stripped of
market-wide, sector, and country trends. If Apple went up but the whole tech
sector went up more, Apple scores low. It is binned into 5 values:
`0, 0.25, 0.5, 0.75, 1.0`.

Typical per-era CORR scores land between `0.01` and `0.03`. Scores look tiny
because returns are genuinely noisy and the signal is weak by design.

### Scoring Metrics

**CORR (Correlation):** per-era Pearson correlation between your predictions
and the target. The primary metric. Averaged over all validation eras.

**MMC (Meta Model Contribution):** how uniquely additive your model is to
Numerai's aggregate. A model that copies the crowd scores 0 MMC even if
its CORR is good.

**Sharpe ratio:** mean ÷ std across eras. Consistency matters more than
peak performance. A model with mean=0.02, std=0.01 (Sharpe=2.0) beats
mean=0.03, std=0.06 (Sharpe=0.5) in the long run.

---

## 3. How AutoResearch Comes Into the Picture

### Karpathy's Original Idea

Karpathy's `autoresearch` runs an agent against a GPT training script:
1. Agent modifies `train.py` with one architectural change
2. Trains for a fixed 5-minute wall-clock budget
3. Evaluates on `val_bpb` (bits-per-byte — lower is better)
4. Keeps the change if it improves; reverts via `git reset` if not
5. Commits every winner; logs everything to `results.tsv`
6. Repeats ~12 times per hour, ~100 experiments overnight

The elegance: humans write the research *direction* (`program.md`).
The agent handles the *execution* — code, training, evaluation, bookkeeping.

### Adapting It to Numerai

The financial ML setting requires different design choices:

| Aspect | Karpathy's autoresearch | Our Numerai pipeline |
|---|---|---|
| Metric | val_bpb (lower = better) | CORR Sharpe (higher = better) |
| Time budget | 5-min wall-clock | 30-min target per iteration |
| Editable file | `train.py` (model arch) | `train.py` (features + hyperparams) |
| Evaluation | Fixed eval set | 150 validation eras with 4-era embargo |
| Agent | Claude API | Claude Code CLI (`claude -p`) |
| Revert | `git reset` | `git restore train.py` |

**Key insight for Numerai:** Instead of training a neural network from scratch
each iteration (minutes on a GPU), we retrain LightGBM/XGBoost (minutes on CPU).
The cycle is slower but requires no special hardware.

### Why Claude Code Instead of the API

Instead of calling the Anthropic API directly from a Python script, we use
the `claude` CLI in non-interactive mode:
```bash
claude -p "$(cat loop_prompt.md)" --allowedTools Read,Edit,Write,Bash,Glob,Grep
```

This means the agent already has file editing, git, and bash tools built in.
No custom tool scaffolding needed. The entire orchestration layer is a
50-line shell script.

---

## 4. How the Pipeline Works

### File Structure
```
prediction/
├── train.py          ← ONLY file the agent edits
├── evaluate.py       ← Fixed harness, never edited
├── pipeline.sh       ← Shell loop (runs claude -p in a loop)
├── loop_prompt.md    ← Per-iteration instructions for the agent
├── program.md        ← Human-written research directions
├── results.tsv       ← Experiment log (commit, metrics, status)
└── LESSONS.md        ← Problems encountered and fixes
```

### The Loop (one iteration)

```
1. Agent reads: program.md + results.tsv (last 15 rows) + train.py
2. Determines current best CORR Sharpe from results.tsv
   (or BATCH_BASELINE override from program.md)
3. Proposes ONE focused change not yet tried
4. Edits train.py (updates # EXPERIMENT: comment at top)
5. Runs: python3.10 train.py  →  validation_predictions.parquet
6. Runs: python3.10 evaluate.py  →  metrics.json
7. Decision:
   - Keep if: corr_sharpe improved, OR same score + simpler code
   - Revert if: worse, or crash  →  git restore train.py
8. If keeping: git add train.py && git commit
9. Appends one row to results.tsv
10. Prints: [success] corr_sharpe=1.45 (+0.02) — deeper trees
```

### The Interface Contract

`train.py` can do anything — change the model, features, targets, engineering —
as long as it outputs `validation_predictions.parquet` with columns
`[era, prediction, target]`. `evaluate.py` is fixed and reads only this file.

### Human Steering via `program.md`

The human's role is to update `program.md` between sessions:
- Mark explored ideas as `[x]` so the agent doesn't retry them
- Add new directions based on what worked
- Set constraints (runtime budget, no new pip installs, etc.)
- Update the `BATCH_BASELINE:` score when rolling back

This is the key human-in-the-loop mechanism. The agent handles tactics;
the human handles strategy.

---

## 5. Important Findings

### ML Findings

**The single biggest gain: feature set size (+64% Sharpe)**
Switching from `small` (42 features) to `medium` (780 features) jumped
CORR Sharpe from 0.84 → 1.38 in one step. More features = more signal.
The agent found this on iteration 3.

**More data has diminishing returns**
- Every 4th era → every 2nd era: +2.3% (good)
- Every era (all data): -6.5% (bad — old eras have distribution drift)
Old stock market data is less predictive. Recent data matters more.

**Tree depth sweet spot: max_depth=7, num_leaves=127**
Sequential improvements: depth 5 → 6 → 7 each helped. Depth 8+ not tried.
Going from 31 to 127 leaves enabled richer feature interactions.

**Regularization findings:**
- `subsample=0.8` (row sampling): small win (+0.4%)
- `min_child_samples=200`: slightly hurt (agent over-regularized)
- `colsample_bytree 0.1→0.3`: hurt (0.1 already samples ~78/780 features per tree)

**Target ensembling was the breakthrough**
Training 3 separate models on different targets (`target`, `target_cyrusd_20`,
`target_victor_20`) and averaging predictions pushed CORR Sharpe to **1.550**
— the highest result. Different targets capture different aspects of returns;
averaging reduces variance.

**Per-era mean-centering: no effect**
Subtracting the era mean from predictions made no difference on CORR Sharpe.
The Numerai scoring function already normalizes per era.

**Ranking predictions within era: catastrophic**
Converting predictions to within-era ranks destroyed performance (0.84 → 0.39).
LightGBM's raw output already encodes relative ordering; re-ranking loses
calibration information.

### Pipeline / Engineering Findings

**`tail -f` deadlock:** The agent spontaneously decided to run training as a
background process and monitor it with `tail -f`. `tail -f` never exits on a
completed file — the iteration ran for 1h47min doing nothing.

**macOS vs Linux:** `timeout` (GNU coreutils) is not available on macOS.
Had to replace with a bash watchdog (background `sleep` + `kill`).

**Baseline score drift:** Rolling back `train.py` without updating the reference
score caused all subsequent experiments to appear as failures. The agent was
comparing single-model results against the ensemble's superior score.
Solution: explicit `BATCH_BASELINE:` override in `program.md`.

**Target name precision matters:** `target_cyrus` ≠ `target_cyrusd_20`.
One wrong character in `program.md` caused a crash on the first target experiment.
Lesson: always give the agent exact column names, not approximations.

**Iteration time scales with model count:** 1 model × every-2nd-era ≈ 25 min.
3 models × every-2nd-era ≈ 65 min. The relationship is linear. Runtime budgets
must account for ensembling multipliers.

### The Key Metric Journey

```
Iteration  1:  0.84  ← baseline (LightGBM, small features, 4th era)
Iteration  3:  1.38  ← medium features  (+64%)
Iteration  5:  1.40  ← more estimators  (+1.4%)
Iteration  7:  1.41  ← subsample        (+0.4%)
Iteration  8:  1.43  ← more training data (+1.9%)
Iteration 10:  1.45  ← deeper trees     (+1.7%)
Iteration 12:  1.48  ← deeper trees     (+1.7%)
Iteration 13:  1.55  ← 3-target ensemble (+4.8%)  ← BEST
```

**Total improvement over 13 successful iterations: +84% CORR Sharpe**
From 0.84 → 1.55 with zero human-written model code.
