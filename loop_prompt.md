You are running ONE iteration of an automated ML improvement pipeline for the Numerai tournament.

## Steps — execute them in order

### 1. Read context
- Read `program.md` — research directions and constraints
- Read `results.tsv` — past experiments (focus on the last 15 rows)
- Read `train.py` — current model code

### 2. Determine the current best score
Parse `results.tsv` and find the highest `corr_sharpe` among all rows with `status=success`.
If no successful rows exist yet, treat any result as an improvement (first run establishes baseline).

### 3. Propose ONE change
Based on `program.md` and what has already been tried in `results.tsv`, choose ONE focused idea that
has not been tried yet and is likely to improve CORR Sharpe. Do not stack multiple changes.

### 4. Edit train.py
- Update the `# EXPERIMENT: <description>` comment at the very top of the file
- Apply your one proposed change

### 5. Run training
```
python3.10 train.py
```
**IMPORTANT: run this as a foreground command — do NOT use background execution or `run_in_background=true`.**
If it crashes or raises an exception:
- Restore train.py: `git restore train.py`
- Skip to step 7 with status=crashed and corr_sharpe=0, mmc_sharpe=0

### 6. Run evaluation
```
python3.10 evaluate.py
```
**IMPORTANT: run this as a foreground command — do NOT use background execution.**
Read `metrics.json` to get the results.

### 7. Decide: keep or revert

**Keep** if any of:
- `corr_sharpe` > current best (improvement)
- `corr_sharpe` >= current best AND the new train.py is shorter/simpler (simplification win)

**Revert** if:
- `corr_sharpe` < current best
- Training or evaluation crashed

To revert: `git restore train.py`

### 8. Commit if keeping
```
git add train.py
git commit -m "<one-line description of the change>"
```

### 9. Append one row to results.tsv
Tab-separated columns in this exact order:
`commit_hash  corr_mean  corr_sharpe  mmc_sharpe  max_drawdown  status  description`

- `commit_hash`: output of `git rev-parse --short HEAD`
- `corr_mean`, `corr_sharpe`, `mmc_sharpe`, `max_drawdown`: from metrics.json (6 decimal places)
- `status`: `success`, `failed`, or `crashed`
- `description`: the `# EXPERIMENT:` comment from train.py (without the prefix)

### 10. Print a one-line summary
Example: `[success] corr_sharpe=0.9123 (+0.0361) — switched to medium features`

## Hard rules
- Edit ONLY `train.py` — never touch evaluate.py, pipeline.sh, loop_prompt.md, or results.tsv format
- One change per iteration
- If unsure between two ideas, pick the simpler one
