# Optimization Analysis Playbook

When the user asks to analyze an optimization run, read this file first, then
load the relevant CSVs from `nfelo/Optimizer/results/` and apply the methods
below. The goal is a **qualitative interpretation of the results, grounded in
quantitative analysis** — the numbers are evidence, not the report.

---

## Philosophy — why we tune nfelo the way we do

The NFL market is exceptional at prediction, and especially against objective
functions like Brier which penalize larger errors more. Models can also be 
accurate in prediction, but their usefulness comes more from their opinionated-ness.
Models can have far more variance in prediction and still carry meaningful signal
and alpha in their divergence from the market. This does, however, create a
problem for optimization as opinionated models with strong opinions will
inevitably be penalized more for their mistakes as they are more extreme.

This becomes especially important when tuning a model's market regression, as
the most "accurate" model by any loss function will be the one that regresses
hardest to the stability of market predictions. As a result, nfelo is trained
in two stages.

First, the base model is tuned without any market awareness against
a pure Brier score to get the cleanest read on its ability to simply project and
react mechanically to NFL results. The unregressed model is then passed to a second
optimization that focuses on the pieces of the config that influence market regression.
Rather than focus on pure brier, which would push the model towards mirroring the
market, it considers brier adjusted for correlation with the market and ATS performance.
The goal is to regress true error to the market to score "free" accuracy boosts, while
preserving (ie not regressing) instances where model disagreement proves to be warranted.

In plain terms, the model should regress fully to the market when it has a weak opinion
on a matchup, and regress very little when it has conviction.

The **two stages** in detail:

1. **Stage 1 — Base model.** Tune the model with no market awareness. The
   objective is unregressed Brier (`nfelo_brier`, `nfelo_brier_adj`, etc.). The
   feature set covers core Elo and rating-blend params: `k`, `z`, `b`,
   `reversion`, `dvoa_weight`, `wt_ratings_weight`, `margin_weight`,
   `pff_weight`, `wepa_weight`, `market_resist_factor`. The goal is the best
   market-unaware model we can build. Because we have not yet introduced the
   market, **overfitting avoidance dominates** — a "best Brier" config that
   looks unusual relative to its neighbors is suspect.

2. **Stage 2 — Market factors.** Holding the base model fixed, tune the market
   regression family of features: `market_regression`, `se_span`, `rmse_base`,
   `spread_delta_base`, `hook_certainty`, `long_line_inflator`, `min_mr`.
   Objective is the regressed Brier (`nfelo_brier_close`, etc.). Here the
   pull-to-market force from the loss function is dominant, so the analysis
   has to **trade Brier against alpha preservation** — a configuration that
   maximizes Brier by collapsing the model into the market is a degenerate
   solution.

The same playbook applies to both stages. What differs is how we pick the
"winning" config (see *Picking the config* near the bottom).

---

## Results layout

Each optimization run produces three CSVs in `results/`:

- `{opti_tag}-{opti_date}.csv` — train rows, one per new-best the optimizer found.
- `{opti_tag}-{opti_date}_test.csv` — skinny test row per new-best (joined by `run_id`).
- `{opti_tag}-{opti_date}_benchmarks.csv` — one-time market + market_open metrics for each split.

---

## Conventions

These are non-negotiable. Reversing them produces wrong analysis.

- **Higher Brier is better.** nfelo's Brier-style metric is sign-flipped vs.
  textbook. The optimizer log line `Run number N - -1.484` shows the
  scipy-minimizable form; the saved `achieved_value` and `brier*` columns are
  the real (positive) numbers. Sort descending. Positive delta vs. market =
  model wins.
- **`random_starts=True` always.** Every result we trust comes from random
  starts (the `RandomStarts` strategy). A single local optimization is for
  smoke-tests only. This optimization problem is not smooth.
- **`run_id = {hop_number}-{total_runs}`.** Use it to join train ↔ test ↔ benchmarks.
- **Test sets can be misleading** Small test sets introduce noise and can result in
  wildly positive or negative deltas. Instead of treating a positive delta vs train
  as an explicit indication of OOS generalizability, or a negative delta vs train
  as an explicty indiciation of overfitting, value correlations between train
  and test metrics more than pure delta analysis (Method 2 below).
- **Identify which stage you are analyzing before drawing conclusions.** Stage
  is implied by the `objective` column in the train CSV (`nfelo_brier*` for
  stage 1, `nfelo_brier_close*` for stage 2) and by which features were tuned
  (check the feature set written in the train row vs. the `available_features`
  dict in `nfelo/Optimizer/Primitives/NfeloOptimizerBase.py`). The stage
  changes how you rank "best".

---

## Understand the run before interpreting it

Statistics on a feature you do not understand are noise. Before computing
anything, do this:

1. **Read the train CSV header to identify which features the run actually
   flexed.** The CSV will contain every config column, but only a subset are
   actively tuned -- those are listed in the `features` argument the run
   passes to `NfeloOptimizer` (look at the run's invocation script).
2. **For each flexed feature, learn what it does in the model.** The
   definitions and bounds live at
   `nfelo/Optimizer/Primitives/NfeloOptimizerBase.py:17` in
   `available_features`. The *behavior* — what the feature controls in the
   prediction pipeline — lives in `nfelo/Model/Nfelo.py` (search for the
   feature name to find where it enters the math).
3. **Understand the downstream metrics.** `ats`, `ats_be`, `ats_be_play_pct`,
   `brier_ats_adj`, `market_correl` are produced by `NfeloGrader`. To
   interpret them correctly, read the scoring logic in
   `nfelo/Performance/NfeloGraderModel.py`. In particular, you must know
   exactly how `ats_be` and `ats_be_play_pct` are computed before you can
   reason about the Pareto frontier in stage 2 (units won/lost ≠ raw plays).

The output you produce should reflect this understanding. An analysis that
says "feature X is at the bound" without explaining what X *does* and why
that's a concern (or a feature of the design) is incomplete.

---

## Building the working dataset

Every method below operates on a **deduped best-per-hop dataset** built inline:

```python
import pandas as pd
import pathlib
## pick the most-recently-modified train CSV in results/ ##
results = pathlib.Path('nfelo/Optimizer/results')
train_csvs = [p for p in results.glob('*.csv') if not p.stem.endswith('_test') and not p.stem.endswith('_benchmarks')]
train_path = max(train_csvs, key=lambda p: p.stat().st_mtime)
test_path = train_path.with_name(train_path.stem + '_test.csv')
bench_path = train_path.with_name(train_path.stem + '_benchmarks.csv')
## load all three -- benchmarks is optional ##
train = pd.read_csv(train_path, index_col=0)
test = pd.read_csv(test_path, index_col=0)
bench = pd.read_csv(bench_path, index_col=0) if bench_path.exists() else None
## dedupe to one row per hop -- the row with the highest Brier within each hop ##
hops = train.sort_values('brier_nfelo_close', ascending=False).drop_duplicates(subset='hop_number', keep='first')
## join the test row for each surviving run_id; test cols get prefixed `test_` ##
joined = hops.merge(test.add_prefix('test_').rename(columns={'test_run_id': 'run_id'}), on='run_id', how='inner')
```

`joined` is the working dataset for every method below: one row per hop, with
train columns under their original names (`brier_nfelo_close`, `ats`, ...) and
test columns prefixed `test_` (`test_brier_nfelo_close`, `test_ats`, ...).
Benchmark values for delta calculations live in `bench` (filter by `split` and
`model_name`).

---

## Method 1 — Feature convergence across hops

**Goal.** Decide which features matter.

**Concept** A feature whose optimized value lands in the same place across many random
starts is being meaningfully pinned down by the objective, whereas a feature whose
optimized value is scattered across its range is either irrelevant to the objective
or under-identified by the data.

Additionally, depending on train/test split its likely to see the test results being
consistently worse than the train results, which would typically be an indicator
of overfitting. However, since NFL results can be 1) noisy 2) non-stationary, and 3)
it is hard to beat the benchamrk, this does not alway mean the model is overfitting
to the extent the results woudl suggest. To determine the degree of overfitting,
we also look at the correlation between the train and test results with respect to the objective.
Even if test results are negative, if they correlate with train results, we take that to mean
there is a reasonable mitigation of overfitting.

**Compute.** Across the deduped best-per-hop dataset (one row per hop), for each
feature column:

```python
stats = joined[FEATURES].agg(['mean', 'std', 'min', 'max'])
cv = (joined[FEATURES].std() / joined[FEATURES].mean().abs()).rename('cv')
## bound-hit %: fraction of hops where value is at the configured min or max ##
## bounds live in nfelo.Model.Nfelo.Nfelo.available_features[feat] ##
lo_hit = (joined[feat] <= bounds[feat]['min'] + tol).mean()
hi_hit = (joined[feat] >= bounds[feat]['max'] - tol).mean()
## rank correlation between feature value and train objective ##
corr_obj = joined[[feat, 'brier_nfelo_close']].corr(method='spearman').iloc[0, 1]
```

**Reading it.**

- Low spread / low CV → feature is converging across hops → the objective genuinely
  prefers a particular value → this feature has impact.
- High spread / high CV → optimizer can't pin it down → likely low impact OR
  under-identified (e.g., redundant with another feature). Consider what the feature is
  attempting to accomplish, analyze the code and its results, and determine 1) if their are
  suggestions for how to better achieve the feature's intended goal, 2) reasons
  why the feature is not achieving its goal, or 3) reasons to drop if from the model.
- Bound-hits >20% on one side → either the bound is wrong (widen and re-run) OR
  it's an intentional tradeoff lever (e.g., `min_mr` stuck at 0 = "we want as
  little floor on regression as possible"). Check the known-tradeoffs section
  before assuming the bound is wrong.
- `corr_obj` near 0 with high CV → feature isn't driving the objective at all.
- `corr_obj` strong (|r| > 0.3) and feature converging → feature is doing real work.

**Phrasing template (explanation first, metric as evidence).**

- *"`market_regression` is converging tightly across hops (CV = 0.04, range 0.69–0.78),
  and the objective prefers higher values (Spearman = +0.61) — this is a real lever."*
- *"`min_mr` doesn't converge (CV = 0.92) and 38% of hops land at the lower bound — the
  optimizer doesn't see strong signal from this feature, or 0 is genuinely the
  preferred value and the bound should be removed."*
- *"`se_span` shows no objective correlation (Spearman = +0.04) and bounces across
  its full range — likely not impacting `brier_nfelo_close`."*

---

## Method 2 — Train / test correlation

**Goal.** Distinguish "test deltas are bad because the holdout is noisy" from
"test deltas are bad because we're overfitting train." Absolute deltas can't
tell these apart on 570 games. Correlation can.

**Compute.** On the deduped best-per-hop dataset, join train + test rows by
`run_id`, then:

```python
for m in ['brier_nfelo_close', 'su', 'ats', 'ats_be']:
    test_col = 'test_{0}'.format(m)
    pearson = joined[m].corr(joined[test_col], method='pearson')
    spearman = joined[m].corr(joined[test_col], method='spearman')
```

**Reading it.**

- Strong positive (|r| > 0.3): configs that win on train also win on test (in
  rank, even if absolute level is lower). Signal is real; test gap is noise.
- Near zero: train ordering doesn't predict test ordering. The optimizer is
  fitting train idiosyncrasies. This is the actual overfit tell.
- Negative: configs that win on train *lose* on test. Strong overfit OR
  market non-stationarity (the test era rewards different behavior than the
  train era — common on recent ATS where the modern market has tightened).
- Brier vs. ATS correlations often differ. Brier (probability calibration) tends
  to track across splits more reliably than ATS (binary cover) because ATS is
  noisier per-game.

**Phrasing template.**

- *"Brier tracks across train and test (Spearman = +0.38), so the consistently
  negative test delta is more plausibly small-sample variance than overfit."*
- *"ATS performance doesn't correlate train-to-test (Spearman = +0.06) — our ATS
  tuning is finding train noise, not durable edge."*

---

## Method 2b — Regression efficiency (Brier ↔ ats_be)

**Goal.** Distinguish "Brier is up because the model is regressing toward the
market" (degenerate) from "Brier is up because the model is shedding noise
plays while preserving signal plays" (efficient). This is a sharper
overfit-detection signal than Method 2 alone.

**Concept.** When the model regresses harder (higher Brier), play volume drops
-- this is the known tradeoff. The question is whether the *surviving* plays
get sharper. If `ats_be` (win rate on plays that clear breakeven EV) goes UP
as Brier goes up, the model isn't just collapsing into the market -- it's
shedding the low-conviction noise plays while keeping the high-conviction
signal plays. That's regression efficiency.

**Compute.** On the deduped best-per-hop dataset:

```python
print('train brier vs train ats_be:        {0:+.4f}'.format(joined['brier_nfelo_close'].corr(joined['ats_be'], method='spearman')))
print('test  brier vs test  ats_be:        {0:+.4f}'.format(joined['test_brier_nfelo_close'].corr(joined['test_ats_be'], method='spearman')))
print('train brier vs test  ats_be:        {0:+.4f}'.format(joined['brier_nfelo_close'].corr(joined['test_ats_be'], method='spearman')))
```

The cross-split number (train Brier ↔ test ats_be) is the load-bearing one --
it asks "if I pick the train-best config, do I get sharper bets on test too?"

**Reading it.**

- All three positive and meaningful (|r| > 0.3): real regression efficiency.
  The model is making *better* bets, not just *fewer* bets. Strong
  not-overfitting signal -- stronger than Brier transfer alone.
- Train positive but cross-split near zero or negative: efficiency on train
  is an artifact -- the configs that look sharp on train don't transfer. This
  is the overfit case.
- All three near zero: the relationship between calibration and bet-quality
  is decoupled. Probably means Brier is improving via market regression but
  not via signal preservation.
- Look at the *very top* of the Brier list as a separate check: across the
  full distribution efficiency may be positive, but the literal Brier-max
  configs sometimes invert (squeezing the last drops of Brier by regressing
  some signal-bearing plays). When this happens, the Pareto frontier in
  *Picking the config* will pick configs slightly below the Brier ceiling.

**Phrasing template.**

- *"Regression efficiency is real and transfers: train Brier and train ats_be
  correlate +0.67, test Brier and test ats_be +0.56, and crucially the
  train Brier predicts test ats_be at +0.50 -- the model is shedding noise
  plays while preserving signal plays, not just collapsing into the market."*
- *"Brier and ats_be are uncorrelated on test (+0.04) -- the train-side
  efficiency doesn't transfer, suggesting the optimizer is fitting train
  ats_be noise."*

---

## Method 3 — Tradeoff diagnostics

**Goal.** Surface (or confirm) tradeoffs between features and downstream metrics
empirically, instead of relying on intuition.

**Compute.** On the deduped best-per-hop dataset, Spearman correlation between
each feature's optimized value and each downstream metric:

```python
for feat in FEATURES:
    for metric in ['brier_nfelo_close', 'ats', 'ats_be', 'ats_be_play_pct']:
        corr = joined[[feat, metric]].corr(method='spearman').iloc[0, 1]
```

**Reading it.**

- A feature with opposing-sign correlations to two metrics is a tradeoff lever.
  Example: `market_regression` positive vs. `brier_nfelo_close` and negative vs.
  `ats_be_play_pct` confirms the known "more market regression → better
  calibration but fewer plays" relationship.
- A feature with same-sign correlations to multiple goal metrics is "free": no
  tradeoff, push in the indicated direction.
- A feature that correlates with nothing is a candidate for removal from the
  feature set.

**Phrasing template.**

- *"`market_regression` trades calibration against play volume: pushing it up
  improves train Brier (Spearman = +0.71) but cuts plays sharply
  (Spearman = −0.82 on `ats_be_play_pct`) — the known tradeoff is empirically
  present in this run."*
- *"`hook_certainty` correlates with neither Brier (+0.04) nor ATS (−0.02) —
  it's not doing useful work in this feature set."*

### Known tradeoffs (living list — add to this as you find more)

- **`market_regression` ↑ → Brier ↑, plays ↓.** Pushing the model toward the
  market improves probability calibration but reduces ATS edge by collapsing
  model-market line gaps. This is the structural tension the whole stage-2
  optimization has to navigate.
- *(add new tradeoffs here as analyses uncover them)*

---

## Method 4 — Cross-run comparison

**Goal.** A single run is hard to judge in isolation. Compare against prior
runs of the **same stage** to see whether this run is real progress.

**Compute.** Enumerate train CSVs in `results/`, group by `opti_tag` prefix
so stage-1 runs only compete with stage-1 runs (and stage-2 with stage-2):

```python
runs = []
for path in pathlib.Path('nfelo/Optimizer/results').glob('*.csv'):
    if path.stem.endswith('_test') or path.stem.endswith('_benchmarks'):
        continue
    df = pd.read_csv(path, index_col=0)
    deduped = df.sort_values('brier_nfelo_close', ascending=False).drop_duplicates('hop_number', keep='first')
    runs.append({
        'run': path.stem,
        'opti_tag': df['optimization_type'].iloc[0] if 'optimization_type' in df.columns else path.stem,
        'objective': df['objective'].iloc[0] if 'objective' in df.columns else None,
        'n_hops': deduped['hop_number'].nunique(),
        'best_brier': deduped['brier_nfelo_close'].max(),
        'median_brier': deduped['brier_nfelo_close'].median(),
    })
leaderboard = pd.DataFrame(runs)
## partition by opti_tag (stage proxy) before ranking ##
for tag, group in leaderboard.groupby('opti_tag'):
    print(group.sort_values('best_brier', ascending=False))
```

**Reading it.**

- Current run's best > all prior bests in the same `opti_tag` → real progress;
  the code or feature-set change since the last run helped.
- Current run's best ≈ prior bests → this run wasn't worth the compute.
- Current run's best < prior bests → regression. Investigate.
- *Median* matters as much as *best*. A higher best with a lower median means
  this run got lucky once; a higher median means the whole landscape improved.
- **Do not compare across `opti_tag` prefixes.** Stage-1 and stage-2 runs
  optimize different objectives over different feature sets; their numbers
  are not comparable.

---

## Method 5 — Config robustness

**Goal.** Distinguish a *durable* top config from a *flukey* one. A config
whose Brier is high because it sits inside a cluster of similarly-strong
configurations is robust. A config whose Brier is high because it found a
narrow peak — surrounded by configs that score much worse — is overfit risk.

**Concept.** In normalized feature space (each feature scaled to [0, 1] using
its `available_features` min/max bounds), measure how dense the top-K
neighborhood is around each candidate config. Many close neighbors among the
top-K → robust. Few close neighbors → isolated peak → flukey.

**Compute (primary — neighborhood density).** For each of the top K configs
by Brier (e.g., K = 10), count how many of the *other* top-K configs sit
within a small radius in normalized feature space.

```python
import numpy as np
K = 10
top = joined.sort_values('brier_nfelo_close', ascending=False).head(K).copy()
## normalize each tuned feature to [0, 1] using its configured bounds ##
norm = pd.DataFrame(index=top.index)
for feat in FEATURES:
    lo, hi = bounds[feat]['min'], bounds[feat]['max']
    norm[feat] = (top[feat] - lo) / (hi - lo)
## pairwise Euclidean distance in normalized feature space ##
points = norm[FEATURES].to_numpy()
dists = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
np.fill_diagonal(dists, np.inf)
## neighbors within radius r (calibrate r per run -- e.g. 0.25 in unit cube) ##
r = 0.25
top['neighbors_within_r'] = (dists < r).sum(axis=1)
top['mean_dist_to_top_K'] = np.where(np.isinf(dists), np.nan, dists).mean(axis=1)
```

**Compute (secondary — jitter robustness).** *Optional, heavier; requires
re-running the model + grader.* For the single best config, perturb each
feature by ±5% of its bound range and observe how Brier degrades. A robust
config degrades gently; a flukey one falls off a cliff. Useful as a tiebreaker
when neighborhood density is ambiguous.

**Reading it.**

- Top config with many neighbors_within_r and low mean_dist_to_top_K → the
  top of the landscape is a plateau, not a spike. Confident pick.
- Top config with few neighbors and high mean_dist_to_top_K → isolated peak.
  Prefer a slightly-lower-Brier config that sits in a denser cluster.
- The whole top-K with no clustering → the objective surface is messy;
  consider a longer run (more `niter`) or different feature set.

**Phrasing template.**

- *"The best config (`run_id 24-44`) sits inside a tight cluster of 6 other
  top-10 configs within 0.25 normalized distance — this looks like a real
  plateau, not a flukey peak."*
- *"`run_id 14-218` has the second-best Brier but is isolated (1 neighbor
  within radius) — the slightly-lower `run_id 2-156` sits in a denser
  cluster (5 neighbors) and is the safer pick."*

---

## Picking the config — stage-aware ranking

The "best" config is not the same in stage 1 and stage 2. Apply the right
ranking after running Methods 1-5.

### Stage 1 (base model, unregressed Brier objective)

We want **the best market-unaware model**. Overfit avoidance is the binding
concern.

1. Take the top-K configs by train Brier (e.g., K = 10).
2. Apply Method 5 to score robustness.
3. Pick the highest-Brier config whose **neighborhood density** is solidly
   above the median across the top-K. If the literal #1 is isolated, drop to
   the highest-Brier config that sits in a cluster.
4. Sanity-check on the held-out test split: the picked config should at
   minimum **not be near the worst** in test Brier. Method 2's train/test
   correlation tells you whether to trust test rank at all.

### Stage 2 (market factors, regressed Brier objective)

The optimizer's pull toward "be the market" is structural — we have to push
back with a Pareto view. The right axis for that pushback is **units won
or lost**, not raw play volume.

> Units = (play volume) × (edge per play over breakeven).
>
> Read `nfelo/Performance/NfeloGraderModel.py` to confirm exactly how
> `ats_be` (hit rate among plays clearing the breakeven threshold) and
> `ats_be_play_pct` (fraction of games that are plays) are defined for this
> codebase. Compute units consistently with those definitions — a typical
> form is `units = ats_be_play_pct * (ats_be - 0.5238)` (where 0.5238 is the
> breakeven rate at standard -110 odds), but verify against the source.

1. For each of the top configs by train Brier, compute units (per the
   formula above, using values as defined in `NfeloGraderModel.py`).
2. Plot or tabulate the (Brier, units) Pareto frontier — the configs that
   are not dominated by any other config on both axes simultaneously.
3. **The user picks from the frontier.** The playbook does not pick for
   them — the right tradeoff between Brier and units is a model-strategy
   decision, not a statistical one. Surface 2-4 candidate configs from the
   frontier with their (Brier, units) coordinates and the qualitative
   character of each ("high accuracy, low volume" vs. "moderate accuracy,
   higher volume").
4. `ats_be_play_pct` stays informational, not a hard floor. A config with
   very few plays may still be on the frontier; it is the user's call
   whether that profile is acceptable.

---

## Output expectations

The output is a **qualitative interpretation** of the run informed by the
quantitative analysis above — not a stat dump.

1. Open with **which stage** this run is and **what it was trying to do**
   (one sentence: "Stage-2 market-factor tune over {features}, objective
   `nfelo_brier_close`, train 2008-2023, test 2024-2025").
2. State the **headline finding** (one sentence: real progress / lucky once
   / regression / overfit).
3. Walk through the methods **interpretation first, metric as evidence**.
   Do not lead with raw numbers.
4. Flag any **known-tradeoff signals** that the run confirmed or contradicted.
5. End with a **specific next action** — widen a bound, drop a feature,
   re-run with different test seasons, adopt a config, run a stage-1 retune
   before stage-2, etc. Tie it to a feature or a bound, not a vague gesture.

Total length: ~20-30 lines. The numbers in parentheses are evidence. A good
analysis demonstrates understanding of what each feature does in the model
and what the objective rewards, not just that the stats were computed.
