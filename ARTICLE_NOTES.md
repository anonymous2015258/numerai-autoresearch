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

---

### 5.1 ML Findings

#### Finding 1: Feature Set Size Was the Single Biggest Lever (+64% in One Step)

The agent's very first productive idea — on iteration 3 — was to switch from
the `small` feature set (42 features) to `medium` (780 features). CORR Sharpe
jumped from **0.84 to 1.38 in a single training run**. No architecture change.
No hyperparameter tuning. Just more input signal.

This is a striking result because it goes against the instinct to start simple.
The `small` set was advertised as having the "highest feature importance" — but
importance in isolation does not mean the same as joint predictive power.
780 features give LightGBM far more combinations to exploit.

What makes this finding valuable: a human experimenter might spend days tuning
model architecture before revisiting feature selection. The agent — given a
clean priority list in `program.md` — went straight there on iteration 3.

**Takeaway:** In tabular ML, feature set breadth often dominates model
sophistication. Explore data breadth before model depth.

---

#### Finding 2: More Training Data Has Diminishing Returns — and Can Hurt

The training data is organized into `eras` (weekly snapshots). The baseline
used every 4th era to save memory. The agent methodically explored this:

| Training data | CORR Sharpe | Change |
|---|---|---|
| Every 4th era (~375 eras) | 1.400 | baseline |
| Every 2nd era (~750 eras) | 1.431 | **+2.3%** ✅ |
| Every era (~1,500 eras) | 1.352 | **-6.5%** ❌ |

The surprising result: using ALL available eras hurt performance. Why?

Numerai's dataset spans ~28 years. The oldest eras (early 2000s) describe a
structurally different stock market — fewer quant funds, different market
microstructure, different factor dynamics. Training on them adds noise that
overwhelms the signal from recent eras.

This is a domain-specific nuance the agent discovered empirically without
any financial knowledge. It tried all three options and the data told the story.

**Takeaway:** More historical data is not always better in financial ML.
Recency bias can be a feature, not a bug.

---

#### Finding 3: Tree Depth Improvements Were Consistent But Hit a Plateau

The agent incrementally deepened the tree structure across four iterations:

| Config | CORR Sharpe | Change |
|---|---|---|
| max_depth=5, num_leaves=31 | 1.405 | starting point |
| max_depth=6, num_leaves=63 | 1.454 | **+1.7%** ✅ |
| max_depth=7, num_leaves=127 | 1.479 | **+1.7%** ✅ |
| max_depth=7, num_leaves=255 | 1.552 | **+0.03%** ✅ (negligible) |

Each doubling of leaves allowed the model to capture more complex feature
interactions — a stock's P/E ratio matters differently depending on its sector,
size, and momentum, and deeper trees can encode these conditional relationships.

The pattern of consistent small gains shows the agent exploring a smooth
optimization landscape. Notably, jumping from 127 → 255 leaves yielded almost
zero gain, suggesting the model had reached its depth saturation point on
this dataset — 780 medium features × 750 training eras can only support so
many distinct leaf conditions before trees start memorizing noise.

Notably: adding `min_child_samples=200` (a leaf-level regularization) at
depth=7 slightly hurt performance. The model at this depth needs flexible
leaf splitting; over-constraining it removes the benefit of the extra depth.

**Takeaway:** Depth improvements in LightGBM are best explored incrementally.
The diminishing returns signal (1.7% → 1.7% → 0.03%) tells you when to stop
and explore other dimensions.

---

#### Finding 4: Stochastic Regularization Helped; Feature Subsampling Did Not

Two regularization experiments had opposite outcomes:

**`subsample=0.8` (row subsampling): +0.4%**
Each tree is built on a random 80% of training rows. This injects noise that
prevents the model from memorizing era-specific patterns — effectively a form
of temporal dropout. Small but consistent win.

**`colsample_bytree=0.3` (feature subsampling): -0.65%**
Increasing the feature sampling ratio from 0.1 to 0.3 hurt. With 780 features,
`colsample_bytree=0.1` already samples ~78 features per tree — a rich enough
set. Increasing to 0.3 (~234 features) forces trees to consider redundant
features, adding noise without adding signal.

The lesson: feature subsampling at `colsample_bytree=0.1` is already aggressive
enough for a 780-feature dataset. Row subsampling addresses a different
problem (temporal overfitting) and remains beneficial.

---

#### Finding 5: Target Ensembling Was the Breakthrough — and the 2nd Target Did Most of the Work

The biggest single jump after the feature set expansion came from ensembling
across multiple targets. Instead of training one model to predict `target`,
the agent trained three separate LightGBM models:
- `target` — the standard 20-day return
- `target_cyrusd_20` — a different definition of 20-day stock-specific returns
- `target_victor_20` — another variant

Final prediction = average of all three model outputs.

Result: CORR Sharpe **1.479 → 1.550** (+4.8%). But later isolated runs revealed
how much each target was contributing individually:

| Ensemble | CORR Sharpe | vs single-model |
|---|---|---|
| `target` only | 1.479 | baseline |
| `target` + `target_cyrusd_20` | **1.553** | **+5.0%** |
| `target` + `target_victor_20` | 1.488 | +0.6% |
| `target` + `target_cyrusd_20` + `target_victor_20` | 1.550 | +4.8% |

The 2-target ensemble nearly matched the 3-target result at 33% lower compute
cost. `target_victor_20` contributed almost nothing — its signal appears
largely captured by `target_cyrusd_20` already. This is a critical insight:
not all targets are orthogonal. The right combination matters more than the
quantity of targets.

**Why this works:** Each target measures future returns with different factor
neutralization and smoothing. A model trained on `target_cyrusd_20` learns
patterns that a `target`-trained model misses. Averaging their predictions
reduces prediction variance while preserving the signal each captures.

**The cost:** Training time scales linearly with target count.
1 model ≈ 25 min, 3 models ≈ 65 min per iteration. The 3-target approach
caused several stalled iterations and was ultimately rolled back.
The 2-target config with `DOWNSAMPLE_EVERY_N_ERAS=4` restored
manageable iteration times (~25 min) with nearly the same CORR Sharpe.

**Takeaway:** Target ensembling is highly effective, but test targets in
isolation before combining. Two complementary targets often outperform
three correlated ones — and cost a third less to train.

---

#### Finding 6: Era-Level Prediction Transformations Were Largely Ineffective

Two post-processing transformations were tested after model training:

**Ranking predictions within era: catastrophic (−54%)**
This was the agent's very first proposed change after establishing baseline.
Converting raw model scores to within-era ranks dropped CORR Sharpe from
0.84 to 0.39 — nearly half the signal destroyed.

Why? LightGBM's raw outputs already encode the correct relative ordering.
Rank-transforming them discards the *magnitude* of predicted differences
between stocks. A stock predicted at 0.52 vs one at 0.50 represents a
meaningful signal difference; both becoming adjacent ranks eliminates this.
Numerai's `numerai_corr` scoring function already handles cross-era
normalization — applying it manually beforehand is double-normalization.

**Per-era mean-centering: no effect**
Subtracting each era's mean prediction before scoring had zero impact.
Again, Numerai's scoring is already per-era — there is nothing to remove.

Both findings point to the same principle: do not pre-process predictions
in ways that replicate what the scoring function already does.

---

### 5.2 Pipeline & Engineering Findings

#### Engineering Finding 1: The `tail -f` Deadlock (1h 47min Lost)

This was the most dramatic failure of the entire run.

The agent was asked to run `python3.10 train.py` to train a model. Instead of
running it as a foreground process (which blocks until done and returns output),
the agent decided to run it as a **background process** and monitor its output
with `tail -f`:

```bash
python3.10 train.py &
tail -f <output_file>
```

Training completed normally — the last line of output read:
```
Validation predictions saved: 1,966,115 rows across 316 eras
```

But `tail -f` does not exit when a file stops being written to.
It polls indefinitely, waiting for more output that will never come.
The agent was stuck — waiting for its own monitoring tool to finish.

The child process had accumulated **1 minute 55 seconds of CPU time over
1 hour 47 minutes of wall time** — 98% of the iteration was spent doing nothing.

Discovery: checking `ps aux` showed no `python3.10` process (training had
finished), but the parent `claude` subprocess had a child:
```
/bin/zsh -c ... tail -f <task_output_file>
```
Killing that child shell immediately unblocked the parent and the pipeline
continued to the next iteration.

**Root cause:** The agent's Bash tool supports `run_in_background=true` for
long-running commands. The agent used it for training, correctly anticipating
that training would be slow. But `tail -f` monitoring a completed file is an
infinite wait. The fix was a one-line instruction in `loop_prompt.md`:
```
IMPORTANT: run this as a foreground command — do NOT use background execution.
```

**Lesson:** When giving agents long-running shell commands, explicitly specify
foreground execution. Agents will reasonably try to be "helpful" by running
things in the background — but the monitoring pattern they reach for (`tail -f`)
has no natural exit condition.

---

#### Engineering Finding 2: The `timeout` Command Does Not Exist on macOS

After the `tail -f` incident, the obvious fix was a time limit on each
iteration. Adding `timeout 3600 claude ...` to `pipeline.sh` seemed straightforward.

The next run output:
```
./pipeline.sh: line 81: timeout: command not found
[pipeline] iteration 1 timed out or errored — continuing
[pipeline] iteration 2 timed out or errored — continuing
```

Both iterations completed in under 5 seconds. No training ran at all.

`timeout` is a GNU coreutils command — standard on Linux, absent on macOS's
BSD-derived shell. The `|| echo "[pipeline] timed out"` safety net meant the
pipeline silently swallowed the error and moved on, making it look like two
successful (but empty) iterations.

**Fix:** A macOS-native watchdog using pure bash:
```bash
"$CLAUDE_BIN" ... &
CLAUDE_PID=$!
( sleep 3600; kill "$CLAUDE_PID" 2>/dev/null ) &
WATCHDOG_PID=$!
wait "$CLAUDE_PID" || true
kill "$WATCHDOG_PID" 2>/dev/null || true
```
This spawns a background timer that kills the main process after 60 minutes
if it is still running. No external dependencies.

**Lesson:** Never assume GNU coreutils availability when writing shell scripts
for macOS. `timeout`, `date -d`, `readlink -f`, and `sed -i` all behave
differently or are absent. Test on the target platform.

---

#### Engineering Finding 3: Baseline Score Drift After Rollback

We rolled back `train.py` from the 3-target ensemble (CORR Sharpe 1.550,
but slow) to the single-model baseline (1.479) to get faster iterations.

What we forgot: `results.tsv` still contained the ensemble's 1.550 row
as a `success`. The agent's loop_prompt.md said:
> "find the highest corr_sharpe among rows with status=success"

So for every subsequent experiment, the agent was comparing against 1.550.
Every single-model experiment scored 1.47–1.48 — all correctly logged as
`failed`. Two full iterations were wasted before we noticed.

The symptom was subtle: results.tsv showed valid-looking metric values,
just marked `failed`. Without reading the description column it looked like
the experiments were genuinely not improving.

**Fix:** Added a `BATCH_BASELINE:` key to `program.md` and updated
`loop_prompt.md` to check for it:
```
BATCH_BASELINE: 1.4786
(The 1.550357 in results.tsv was from a slow ensemble — rolled back.
Compare against 1.4786.)
```

**Lesson:** When a research pipeline allows rollbacks, the scoring reference
must be explicitly managed. The `results.tsv` log of historical scores is not
the same as the "current working baseline." Decouple them.

---

#### Engineering Finding 4: Target Column Name Precision

`program.md` listed a target to explore as `target_cyrus`. The actual column
name in Numerai v5.2 is `target_cyrusd_20`. One character difference.

The agent read `program.md`, used `target_cyrus` literally in `train.py`,
and `train.py` crashed with a `KeyError` on the first target experiment.
Logged as `crashed` with `corr_sharpe=0.000000` in `results.tsv`.

The iteration was not retried — the agent moved on and the correct target
went unexplored for multiple batches.

**Fix:** Updated `program.md` with exact, copy-pasteable column names for
all 41 targets, sourced directly from `features.json`.

**Lesson:** `program.md` is not prose — it is instructions that will be
executed verbatim. Every name, path, and parameter in it should be exact.
Approximations that a human would understand ("target_cyrus ≈ target_cyrusd_20")
will break an agent that does not autocorrect.

---

#### Engineering Finding 5: The claude Binary Is Not on the System PATH

`pipeline.sh` called `claude` directly. When invoked as a subprocess from
another process (rather than from an interactive shell), the `$PATH` does not
include VS Code extension directories.

```
ERROR: 'claude' CLI not found. Install Claude Code first.
```

Claude Code is bundled inside the VS Code extension, not installed as a
system binary. The binary lives at a path like:
```
~/.vscode/extensions/anthropic.claude-code-2.1.86-darwin-x64/
    resources/native-binary/claude
```

**Fix:** Auto-discover the latest version at runtime:
```bash
CLAUDE_BIN=$(
    find "$HOME/.vscode/extensions" -path "*/native-binary/claude" -type f \
    | sort -V | tail -1
)
```

This finds all installed Claude Code versions, sorts semantically, and picks
the latest. If Claude Code is also on PATH (e.g., installed via `npm install -g`),
that takes priority.

**Lesson:** Subprocesses do not inherit interactive shell PATH. Always resolve
tool binaries to absolute paths in automation scripts.

---

### 5.3 Additional Findings from Batch 3 (Runs 16–21)

With the pipeline stable and the baseline reset to the single-model config
(1.4786), the agent continued exploring — this time with the 2-target
constraint and the `BATCH_BASELINE` override in place.

#### Finding 7: Tighter Feature Subsampling Hit a New Overall Best

Earlier, increasing `colsample_bytree` from 0.1 to 0.3 had hurt performance.
The agent now tried the opposite direction — tightening it from 0.1 to **0.05**
(sampling ~39 of 780 features per tree instead of ~78).

Result: CORR Sharpe **1.553 → 1.561** — a new overall best.

Why does sampling fewer features help? With 780 features, the most predictive
ones dominate at `colsample_bytree=0.1`. Tightening to 0.05 forces each tree
to explore more diverse feature subsets, creating an ensemble of trees with
lower internal correlation — which reduces prediction variance. It is, in
effect, a second layer of diversity on top of the target diversity.

The pattern is U-shaped: too many features per tree (0.3) adds noise;
too few (0.1 was already considered low) prevents good splits. The optimum
for this dataset appears to be around 0.05–0.10.

#### Finding 8: Evaluation Methodology Affects Apparent Scores

One iteration changed the validation evaluation itself — instead of evaluating
on every 4th validation era (~75 eras), it evaluated on all validation eras
(~315 eras).

The score dropped from 1.561 to 1.527. This is not a regression — it is a
**methodological change**. With more eras, the Sharpe estimate is more stable
(lower variance) but the score appears lower because lucky high-scoring eras
are diluted by average ones.

This is an important nuance for interpreting results: scores from different
evaluation subsets are not directly comparable. The 1.527 on 315 eras may
actually represent a more reliable model than 1.561 on 75 eras.

#### Finding 9: L2 Regularization Gave a Small Further Gain

Adding `reg_lambda=1.0` (L2 regularization on leaf weights) to the 2-target
ensemble with full validation gave CORR Sharpe **1.527 → 1.534**.

L2 regularization penalizes large leaf weights, pushing the model toward
smoother, more conservative predictions. In Numerai's noisy financial setting,
this slight constraint on prediction magnitude tends to improve generalization.

---

### 5.4 The Key Metric Journey (All 21 Runs)

```
Iter  1:  0.8415  ← baseline: LightGBM, small features (42), every 4th era
Iter  3:  1.3808  ← medium features (780)                [+64.1%] ★ biggest jump
Iter  5:  1.4001  ← n_estimators 5000, lr 0.005          [+1.4%]
Iter  7:  1.4056  ← subsample=0.8, subsample_freq=1      [+0.4%]
Iter  8:  1.4313  ← every 2nd era (2× training data)     [+1.8%]
Iter 10:  1.4544  ← max_depth=6, num_leaves=63           [+1.6%]
Iter 12:  1.4786  ← max_depth=7, num_leaves=127          [+1.7%]  ★ best single model
Iter 13:  1.5504  ← 3-target ensemble                    [+4.8%]
        --- rollback to single model, reset batch baseline ---
Iter 16:  1.5527  ← 2-target ensemble (target+cyrusd_20) [+5.0% vs single]
Iter 17:  1.5522  ← num_leaves=255                       [+0.03%] plateau
Iter 19:  1.5615  ← colsample_bytree=0.05                [+0.6%]  ★ best overall
Iter 20:  1.5266  ← full validation eval (315 eras)      [different baseline]
Iter 21:  1.5344  ← + L2 regularization                  [+0.5% on full eval]
```

**Experiments that failed (8 total):**
- Ranking predictions within era: −54% (catastrophic, do not retry)
- colsample_bytree 0.1→0.3: −0.7%
- Using all eras for training (no downsampling): −5.6%
- min_child_samples=200: −0.2%
- target_cyrus (wrong column name): crashed
- Per-era mean-centering: 0% (no effect)
- `target` + `target_victor_20` only: +0.6% (minimal, victor adds little)
- XGBoost + LightGBM ensemble: inconclusive (compared against wrong baseline)

**Overall:**
- 13 successful improvements, 8 failures out of 21 experiments (62% hit rate)
- Best CORR Sharpe: **1.5615** (2-target ensemble, colsample_bytree=0.05)
- Total gain: **+85.5% CORR Sharpe** (0.8415 → 1.5615)
- Zero lines of model code written by a human
