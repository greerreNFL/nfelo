# Training Playbook

Parallel optimization runs live under `training/runs/`. Each run produces
analysis CSVs at the **run root**; orchestration metadata lives in
`run_details/`. Analysis itself still follows `nfelo/Optimizer/ANALYSIS_PLAYBOOK.md`
— point it at the merged train CSV in the run root, not at individual shards.

---

## Starting a run

```python
from training.NfeloTraining import run_training

run_training('training/runs/my-run/run_details/plan.json')
```

Or build a plan in Python and call `NfeloTraining(plan).run()`.

Requires `PYTHONPATH=.` from the repo root (same as running nfelo scripts).

---

## plan.json

Written to `{run_dir}/run_details/plan.json`. Minimal example:

```json
{
  "run_id": "nfelo-core-2026-06-12",
  "n_shards": 8,
  "repo_root": "/path/to/nfelo/repo",
  "output_root": "/path/to/nfelo/repo/training/runs",
  "opti_tag": "nfelo-core",
  "opti_date": "2026-06-12",
  "features": ["k", "z", "b", "reversion", "dvoa_weight", "wt_ratings_weight", "margin_weight", "pff_weight", "wepa_weight"],
  "objective": "nfelo_brier",
  "test_seasons": [2024, 2025],
  "environment": {
    "type": "local",
    "max_workers": 8,
    "worker_env": { "OMP_NUM_THREADS": "1" }
  }
}
```

| Field | Notes |
|-------|-------|
| `run_id` | Folder name under `output_root` |
| `n_shards` | Independent random-start SLSQP hops |
| `test_seasons` | Omit or `null` for train-only; set to get `_test.csv` |
| `max_seconds_per_shard` | Optional wall-clock cap per shard |
| `environment.max_workers` | Concurrent shards (local subprocess pool) |

Stage presets (`nfelo-core`, `nfelo-base`, `nfelo-mr`) mirror
`nfelo/Development/optimization.py` — same features and objectives.

---

## Output layout

```
training/runs/{run_id}/
  {opti_tag}-{opti_date}.csv              # merged train rows
  {opti_tag}-{opti_date}_test.csv         # when test_seasons set
  {opti_tag}-{opti_date}_benchmarks.csv   # market + market_open per split
  {opti_tag}-{opti_date}_runtime.csv      # per-eval timing log
  run_details/
    plan.json
    summary.json                          # finished / failed counts
    manifest.csv                          # per-shard artifact checklist
    shards/
      shard_001/
        shard.json
        shard_meta.json
        stdout.log
        stderr.log
        {opti_tag}-{opti_date}.csv        # shard-local optimizer output
        ...
```

**Run root = analysis CSVs only.** Everything else is operational detail.

---

## Conventions (shared with optimizer)

- **`run_id` in CSVs = `{hop_number}-{eval_number}`** — for parallel training,
  `hop_number` is set to `shard_id` so rows are unique across shards.
- **Higher Brier is better** (sign-flipped vs textbook). See ANALYSIS_PLAYBOOK.
- **`random_starts` equivalent** — each shard is one fresh random start in
  normalized `[0,1]` space, same as `RandomStarts`.
- **Benchmarks `model_name`**: `market` = market close, `market_open` = open.

---

## Debugging a shard

```bash
cd /path/to/repo
PYTHONPATH=. python -m training.Primitives.Runner \
  training/runs/{run_id}/run_details/shards/shard_003/shard.json
```

Check `stdout.log` / `stderr.log` in the shard directory on failure.

---

## After the run

1. Read `run_details/summary.json` for failed shards.
2. Analyze `{opti_tag}-{opti_date}.csv` at the run root using ANALYSIS_PLAYBOOK.
3. Join test rows on `run_id` when `_test.csv` exists.

Partial success: finished shards are merged even if others fail. Re-run only
failed shard IDs or start a new `run_id`.

---

## Local environment

`environment.type = local` spawns one subprocess per shard via
`python -m training.Primitives.Runner`. Set `OMP_NUM_THREADS=1` per worker
so N shards use N cores without oversubscription.

Daytona / remote environments are not implemented yet; the `Environment`
registry in `training/Environments/` is the extension point.
