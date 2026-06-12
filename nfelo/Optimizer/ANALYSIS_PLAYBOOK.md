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

The rest of this section is a reference: a feature dictionary (what each
tuned parameter does mechanically), a feature-group catalog (which
parameters compensate or co-move and what that means in plain English),
and a methodological note on how feature groups change how you interpret
Method 1's marginal distributions.

---

## Feature dictionary

Every parameter the optimizer can flex, what it controls, and what
happens when it moves up or down. Bounds are from
`Optimizer/Primitives/NfeloOptimizerBase.py:available_features`.

### Core elo dynamics

**`k`** (bound `[5, 20]`) — base shift magnitude per game. Multiplies
the per-game elo update in `calc_shift`. Higher k → larger reactions to
each game's outcome → ratings move faster but become more noise-sensitive.
Lower k → ratings stay close to their prior state → smoother but slower
to incorporate new information.

**`z`** (bound `[200, 600]`) — elo→WP scaling constant. Used in
`elo_to_prob` as `WP = 1 / (10^(-elo_dif/z) + 1)`. Higher z → shallower
sigmoid → moderate elo dif produces moderate WP (close to 0.5).
Lower z → steeper sigmoid → same elo dif produces more confident WP.
Effectively the temperature of the model's probability outputs.

**`b`** (bound `[3, 10]`) — log base in the shift magnitude curve.
Used in `calc_shift` as `shift_magnitude = log_b(|prediction_error|+1) * adj_k`.
Higher b → log grows slowly → tail errors don't blow up shifts (compressed
tails). Lower b → log grows faster → tail errors amplify the shift (heavy
tails).

### Offseason regression

**`reversion`** (bound `[0.0, 1.0]`) — weight on mean reversion to the
league-average elo (1505). At season start, each team's elo is blended
toward the league mean by this fraction. Higher → teams reset toward
average each year. Lower → teams retain their ending elo.

**`dvoa_weight`** (bound `[0.15, 0.5]`) — weight on DVOA preseason
projection in the offseason elo blend. Higher → trust DVOA's projection
of next season more, lean on prior elo less.

**`wt_ratings_weight`** (bound `[0.05, 0.5]`) — weight on the weighted
preseason rating in the offseason elo blend. Higher → trust the weighted
preseason rating more, lean on prior elo less. (Note: `dvoa_weight` and
`wt_ratings_weight` together compete for the offseason blend; they
behave as substitutes — see Feature Groups below.)

### Score assessment (what counts as "what happened")

`calc_weighted_shift` blends three signals to compute the actual elo
update. The weights sum to roughly 1 and trade off against each other.

**`margin_weight`** (bound `[0.1, 1.0]`) — weight on raw scoreboard
margin. Higher → updates driven by the final score itself; outcome-
literal.

**`wepa_weight`** (bound `[0.1, 1.0]`) — weight on weighted EPA
(WEPA). Higher → updates driven by play-by-play efficiency; tries to see
through scoreboard noise.

**`pff_weight`** (bound `[0.1, 1.0]`) — weight on PFF film grade.
Higher → updates driven by per-snap film evaluation; tries to see beyond
both score and EPA.

### Shift modifier

**`market_resist_factor`** (bound `[1.15, 10]`) — inflates `k` in
`calc_shift` when model and market disagreed and the market was right.
Higher → model "resists" disagreeing with the market by penalizing wrong-
direction model-vs-market gaps with larger elo corrections. Lower (toward
1.15) → model trusts itself, accepts model-vs-market disagreement without
extra penalty.

### Market regression (stage 2, MR)

These only flex in stage-2 runs. They control how aggressively the
prediction is blended toward market closing/opening lines.

**`market_regression`** (bound `[0, 0.9]`) — base regression strength.
Multiplies all the per-game regression-factor adjustments. Higher →
model output gets pulled harder toward the market line.

**`spread_delta_base`** (bound `[1.1, 5.0]`) — exponential decay rate
in `initial_mr_factor`. Higher → regression factor decays faster as
|model_line − market_line| grows → model gets to "keep" its opinion when
it strongly disagrees. Lower → regression stays strong even on big model-
vs-market gaps.

**`rmse_base`** (bound `[2, 10]`) — scaling of the rmse-based amplitude
modifier. Higher → recent prediction-error history affects regression
more strongly. Lower → muted effect.

**`se_span`** (bound `[2, 16]`) — window size for the rolling SE used
in `rmse_adj`. Higher → smoother, slower-adapting SE estimate. Lower →
faster but noisier.

**`hook_certainty`** (bound `[-0.5, 0]`) — reduces regression on half-
point ("hook") market lines. Larger magnitude (more negative) → bigger
reduction → model keeps its opinion on hook lines.

**`long_line_inflator`** (bound `[0, 0.75]`) — boost regression on
long-line favorites. Higher → more aggressive regression to market on
big lines.

**`min_mr`** (bound `[0, 0.5]`) — minimum allowed regression factor.
A floor that prevents the multiplicative adjustments from collapsing the
regression to zero. Higher → always at least this much regression.

---

## Feature groups

Many parameters touch the same underlying lever in the model. They tend
to **co-move** (positively correlated across hops) when they jointly
push the same direction, or **compensate** (negatively correlated) when
they substitute for each other.

When parameters belong to a group, the **marginal distribution of any
single one can look wide even though the group's joint behavior is tight**.
A high-k hop might also be a high-z, high-b hop because they're moving
together; another hop with low-k, low-z, low-b sits at a different point
on the *same* group axis but achieves similar objective values. Method 1
will report wide CVs on each individually, when the real story is "the
group has two valleys."

Each group below lists the constituent features, the direction of their
correlation, evidence strength (which runs supported it and at what n),
and a plain-English description of what increasing the group score
means for the model.

### Group: ELO intensity (k, b, z) — *core dial for outcome reactivity*

Members and within-group correlations:
- `k` ↔ `b`: positive (+0.58 pooled, 4/4 runs n=62 total) — both control
  how much each game's update can shift the rating
- `k` ↔ `z`: positive (+0.49 pooled, 4/4 runs n=62 total)
- `z` co-moves because steeper WPs naturally pair with bigger updates;
  flatter WPs pair with smaller updates

**High group score** = aggressive elo dynamics: large per-game shifts,
shallow probability sigmoid, room for tail errors to amplify updates.
The model reacts strongly to new information and forms confident opinions
on game-to-game changes.

**Low group score** = conservative elo dynamics: small per-game shifts,
steeper sigmoid, compressed tail errors. The model is sluggish to update
but produces sharper WP discrimination from stable elo difs.

Both regimes can achieve similar objective; the optimizer finds two
valleys (high-intensity and low-intensity) that score nearly the same.

**Score formula** (each on [0,1] normalized by bound):
```
elo_intensity = mean(
    (k − 5) / 15,
    (b − 3) / 7,
    (z − 200) / 400,
)
```

> Variant: **ELO intensity (update side only)** — drop z, keep `k` and
> `b`. Useful when you want to isolate "how reactive are the elo updates
> themselves" without conflating with WP confidence.

### Group: Preseason priors share (dvoa_weight, wt_ratings_weight) — *fraction of prior elo that gets replaced by external preseason ratings*

This group is **additive, not substitutable**. Per
`offseason_regression`, the model treats `dvoa_weight + wt_ratings_weight`
as a single budget — the total share of prior elo that gets replaced by
external preseason ratings. If the two sum to over 1, they are
renormalized to sum to exactly 1 (100% replacement, no prior elo
retained). Whatever's left over (`1 - dvoa_weight - wt_weight`) goes to
the mean-reverted previous elo.

So: if dvoa AND wt_ratings are both high (sum ≥ 1), the team's starting
elo is fully reset to a DVOA+wt blend. If they sum < 1, the team
partially retains last year's elo. The two features behave as
*complementary contributions to the same priors budget*, not as
substitutes.

The mild negative correlation we observed between the two
(−0.31 to −0.43 in base runs) is consistent with this — within a
fixed total budget, more weight on dvoa pushes weight on wt_ratings
down. But the sum is what determines model behavior; the split between
the two is a secondary choice about *which* preseason signal to lean on.

Members and within-group correlations:
- `dvoa_weight` ↔ `wt_ratings_weight`: negative (−0.31 to −0.43 in base
  runs) — fixed-budget tradeoff within the priors share

**High group score** = preseason priors fully replace previous elo;
teams effectively start the season at their DVOA/wt-ratings projection.

**Low group score** = preseason priors barely matter; the team keeps
their ending elo from last year (subject to the separate `reversion`
mean-revert dial).

Strong inverse relationship with **ELO intensity** (−0.27 pooled on
z↔dvoa across 4 runs; −0.25 on k↔dvoa). When external priors do more
work at season start, in-season elo dynamics need to do less (low
intensity). When priors do little, in-season updates carry the whole
load (high intensity). This is the dominant "Bayesian vs frequentist"
axis of the model.

**Score formula:**
```
preseason_priors_share = min(1.0, dvoa_weight + wt_ratings_weight)
```

> Secondary descriptor (which signal source): `dvoa_share = dvoa_weight /
> (dvoa_weight + wt_ratings_weight)`. Tells you whether the priors share
> is DVOA-tilted (≈1) or wt_ratings-tilted (≈0). Only meaningful when
> the priors share itself is non-trivial.

### Group: Mean reversion (reversion) — *prior pull toward league average*

Single-feature group, but logically distinct from preseason trust.

**High** = teams collapse toward league-average each offseason; elo
identity from one year barely carries over.

**Low** = teams retain ending elo; offseason is just a small adjustment.

In all observed runs, `reversion` strongly prefers low values (high
lo-hit % near 0). The optimizer has consistently found that letting elos
persist across seasons beats forcing them to collapse.

**Score formula:** `reversion_strength = reversion / 1.0`

### Group: Outcome signal mix (margin_weight, wepa_weight, pff_weight) — *what counts as "what happened"*

Members and within-group correlations:
- `margin_weight` ↔ `wepa_weight`: positive (+0.23 pooled, 4/4 runs) —
  both objective game-events
- `pff_weight` is the residual; the three weights together sum to
  roughly 1

This group is multidimensional — it's not just "more weight on outcome
signal" but *which* signal type the model trusts most. Three regime
descriptors:

**Score formulas** (these three sum to about 1 by construction):
```
total = margin_weight + wepa_weight + pff_weight
score_margin = margin_weight / total
score_wepa   = wepa_weight / total
score_pff    = pff_weight / total
```

- **High `score_margin`** = "scoreboard-literal" model — trusts the
  final score
- **High `score_wepa`** = "process-driven" model — trusts play-by-play
  efficiency over the final score
- **High `score_pff`** = "film-driven" model — trusts per-snap quality
  assessment

The optimizer's persistent preference across runs has been
`score_margin` largest, `score_pff` smallest. But the absolute levels
shift between runs — the more useful question is whether your config's
mix is similar to historically-good configs.

### Group: Market-vs-self (market_resist_factor) — *who breaks ties between model and market*

Currently only one feature in this group at the in-season elo update
stage. In stage 2 (MR) the family of market-regression params plays the
analogous role at the prediction stage.

Suggestively in our two in-flight base runs (n=8 and n=9 each, so
preliminary): `market_resist_factor` positively correlates with `z`
(+0.45 pooled) and `b` (+0.64 pooled) — i.e., "trust market more"
coalition pairs with "react more aggressively in elo space when shown
to be wrong." Both pull elo updates the same direction.

**High `market_resist_factor`** = the model penalizes itself heavily
for disagreeing with the market when the market was right; eventually
forces alignment.

**Low `market_resist_factor`** (toward bound 1.15) = the model trusts
itself even when wrong, doesn't extra-penalize model-vs-market gaps.

Both core and base runs consistently push this feature toward the
lower bound — empirically, less market resistance has performed
better. Don't take this at face value: it could mean (a) the model
actually is independently good enough not to need market correction,
or (b) the bound is wrong and should be lowered further to see what
happens.

**Score formula:** `market_resistance = (market_resist_factor − 1.15) / 8.85`

### Group: Market regression strength (market_regression, spread_delta_base, min_mr) — *how hard to pull predictions toward market lines (stage 2)*

Members and within-group correlations (from narrow-retune, n=100):
- `market_regression` ↔ `min_mr`: negative (−0.51 partial)
- `market_regression` ↔ `spread_delta_base`: negative (−0.45 partial)
- `spread_delta_base` ↔ `min_mr`: positive (+0.43 partial)

Interpretation: when `market_regression` (the base strength) is high,
neither `min_mr` (floor) nor `spread_delta_base` (decay aggressiveness)
needs to do work — the base regression already does the job. The three
are substitutes for "how much regression overall."

**Score formula** (combine base strength with inverse of the limiters):
```
mr_strength = mean(
    market_regression / 0.9,
    1 - (spread_delta_base - 1.1) / 3.9,
    min_mr / 0.5,
)
```

**High** = predictions get strongly pulled toward the market; less
divergence from market consensus.

**Low** = predictions retain more model conviction; allowed to disagree
with the market.

### Group: Market regression hooks (hook_certainty, long_line_inflator) — *special-case adjustments*

Members:
- `hook_certainty`: reduces regression on half-point lines (negative
  values mean larger reduction)
- `long_line_inflator`: boosts regression on big lines

These are situational levers, not a unified "more or less regression"
axis. They should mostly be evaluated on their own.

### Group: Adaptation speed (se_span, rmse_base) — *how quickly the model's accuracy-trust adapts*

Members:
- `se_span`: rolling SE window — higher = slower-adapting, smoother
- `rmse_base`: amplitude of the SE-based adjustment

Higher `rmse_base` with lower `se_span` = aggressive accuracy tracking.
Lower `rmse_base` with higher `se_span` = stable, slow-changing accuracy
trust.

This group hasn't shown strong within-group correlation in our runs
(both narrow-retune values were moderate), but mechanistically they're
both about accuracy-history adaptation.

---

## Group score thresholds — grounded in observed configs

Bound-relative thresholds (e.g., "high = > 0.7 of bound range") aren't
useful for narrative comparison because the optimizer doesn't sample the
bound range uniformly. A `reversion` of 0.5 is mid-bound but
*extraordinarily high* relative to any config we've ever observed (the
optimizer typically lo-hits near 0). Thresholds should be cut by
**quantile of observed configs**, not by bound midpoint.

Below are the observed quantile cuts across **all per-hop bests we have
on file** (n=66 for the base/core groups, n=100 for the MR-stage
groups). Use these for both labeling and Method-5 cluster tightness.
Update when adding meaningful new runs.

### Base / core stage groups (pooled from core OLD, core NEW, base OLD, base NEW; n=66)

| group | p10 | p25 | p50 | p75 | p90 |
|---|---|---|---|---|---|
| **elo_intensity** (k,z,b)        | 0.32 | 0.44 | 0.55 | 0.65 | 0.74 |
| **elo_intensity_kb** (k,b only)  | 0.22 | 0.39 | 0.52 | 0.67 | 0.76 |
| **preseason_priors_share**       | 0.45 | 0.51 | 0.65 | 0.72 | 0.80 |
| **mean_reversion**               | 0.00 | 0.01 | 0.07 | 0.29 | 0.42 |
| **score_margin**                 | 0.30 | 0.36 | 0.45 | 0.50 | 0.60 |
| **score_wepa**                   | 0.16 | 0.20 | 0.30 | 0.38 | 0.45 |
| **score_pff**                    | 0.13 | 0.20 | 0.24 | 0.33 | 0.39 |
| **market_resistance** (base only)| 0.04 | 0.04 | 0.04 | 0.10 | 0.44 |

### MR stage groups (pooled from narrow-retune; n=100)

| group | p10 | p25 | p50 | p75 | p90 |
|---|---|---|---|---|---|
| **mr_strength**          | 0.45 | 0.58 | 0.70 | 0.90 | 1.00 |
| **hook_certainty** (raw) | −0.37 | −0.20 | −0.07 | 0.00 | 0.00 |
| **long_line_inflator** (raw) | 0.14 | 0.27 | 0.44 | 0.61 | 0.68 |
| **se_span** (raw)        | 3.8 | 5.6 | 8.5 | 12.0 | 14.6 |
| **rmse_base** (raw)      | 2.1 | 3.3 | 5.6 | 8.0 | 9.0 |

### Reading these tables

The tables above are **descriptive snapshots of where the optimizer has
landed across the runs we have on file**, not category boundaries.
"p75 = 0.65 on elo_intensity" means "of the 66 base/core per-hop bests
we've observed, 25% landed above 0.65" — not "0.65 is high."

The sample is small (66 base/core configs, 100 MR configs, both
dominated by a handful of runs at one moment in the model's life). What
counts as a meaningful "high" or "low" hasn't been validated against
predictive performance, and even the empirical distribution may shift
materially with more runs.

Reasonable uses:
- **Locating a config within history.** "This config's `elo_intensity`
  of 0.72 is above 75% of what we've seen" is a description that
  doesn't pretend to know whether 0.72 is good.
- **Spotting orphan picks.** A picked config that sits far from the
  bulk of the observed distribution (e.g., p < 0.10 or p > 0.90 on
  multiple groups simultaneously) is worth a second look, regardless
  of whether "far from bulk" turns out to be good or bad.
- **Cross-run comparison.** "Run A's top-5 cluster around p60–p70 on
  elo_intensity; Run B's top-5 cluster around p20–p30" is a real
  observation about where each optimizer found its valley.

Uses to avoid:
- Treating p75 as a binary "high" cutoff. The observed distribution
  is one sample; the next major retune may rewrite it.
- Treating tight clustering of observed configs as evidence the
  optimizer has found "the right" region. The optimizer has only
  explored the region it explored.

A few patterns worth noting as observations (not as conclusions):

- `mean_reversion` is heavily concentrated near zero in our observed
  configs (median 0.07, p90 0.42). The optimizer has not, so far,
  pushed toward strong offseason mean reversion.
- `market_resistance` (base stage) similarly clusters near its lower
  bound across both observed base runs. p10=p25=p50 = 0.04.
- `elo_intensity` and `preseason_priors_share` both have observed
  medians somewhat above 0.5.
- `mr_strength` median 0.70 in narrow-retune.

Whether these patterns reflect real model preferences or just the
particular set of runs we've examined is genuinely uncertain.

---

## How to read a config through the group lens

For any single hop's config, compute the seven group scores. The
config's **profile** is the vector of group scores. Two configs with
very different individual parameter values can have *the same* group
profile — and they tend to score similarly on the objective.

Example: a config with `k=18, z=550, b=9.5` (all near upper bounds) and
a config with `k=8, z=300, b=4` (all near lower bounds) have very
different individual values but their `elo_intensity` scores are 0.90
and 0.08 respectively — opposite valleys, both potentially competitive
depending on the rest of the profile.

### Plain-English profile descriptions (orienting language, not categories)

The descriptions below are vocabulary for cross-run comparison, not
classifiers. The combinations they name are *directions* the
combination of group scores can point, not bins a config belongs to.
Use them when narrating "how does this run's optimum differ from
that one's," not to gate decisions.

Stage 1:
- **Reactive + weakly anchored** — high `elo_intensity`, low
  `preseason_priors_share`. The model relies on in-season updates to
  do the work; light preseason priors.
- **Smooth + heavily anchored** — low `elo_intensity`, high
  `preseason_priors_share`. The model leans on preseason DVOA/wt-ratings
  as a strong prior; in-season elo updates are gentle.
- **Hot and well-grounded** — high on both. Uncommon. Often a config
  the optimizer hasn't fully pulled one direction or the other.
- **Underspecified** — low on both. Suspicious for underfitting; not
  enough learning from either source.

Stage 2:
- **Belt and suspenders** — high `mr_strength`, high `market_resistance`.
  Both the elo-update step and the prediction step heavily pull toward
  the market.
- **Opinionated** — low `mr_strength`, low `market_resistance`. Model
  holds its own line at both steps. Higher variance, potentially more
  alpha vs market.

What "high" or "low" means here is **always relative to the configs
you're comparing**, not absolute. Two natural framings:

- *Within a single run*: how does the picked config sit relative to
  the other top-K of its run? Use intra-run quantiles.
- *Across runs*: how does this run's optimum sit relative to history?
  Use the pooled quantile tables above as a description, with the
  caveats spelled out.

### Caveat — thresholds and patterns will drift

The pooled quantile tables were derived from a specific snapshot (5
runs, n=166, dominated by core and base, much of it in-flight at the
time of writing). The shape of the observed distribution is the shape
of the optimizer's search so far, not the shape of "good configs."
Treat the tables as observation, not as ground truth. Recompute when
new major runs land. The script that produced them is the canonical
reference.

---

## Methodological consequences for the other methods

**Method 1 (feature convergence)** marginal CVs are misleading when
features belong to a group. A CV of 0.31 on `k` may not mean "the
optimizer can't pin k" — it may mean "the optimizer found two equally-
good valleys at different `elo_intensity` levels." Always check
group-level convergence before declaring a feature unpinned. If
`elo_intensity` is CV-tight while `k` is CV-wide, the optimizer DOES
have a preference; the feature just doesn't carry it alone.

**Method 5 (top-K neighborhood density)** uses raw Euclidean distance
in normalized feature space, which treats every feature as independent.
For grouped features this penalizes correctly-located but
opposite-valley tops as "isolated." A more honest distance for grouped
features is the distance after mapping features through the group
scores — two configs in the same valley of every group register as
close even if their raw features differ. If Method 5 reports zero
neighbors but the top-K configs share a group profile, that's a real
plateau; the apparent isolation is a metric artifact.

**Method 3 (tradeoff diagnostics)** is more interpretable through group
lens too. A negative `market_resist_factor` ↔ `ats_be` corr looks like
a feature-level tradeoff; through groups, it's part of the broader
"market-aware regime sacrifices high-conviction ATS picks" story.

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

**Phrasing template (explanation first, metric as evidence; calibrate
confidence per the ladder).**

- *"`market_regression` converges tightly across hops (CV = 0.04, range
  0.69–0.78), and higher values correlate with higher objective on this run
  (Spearman = +0.61). Most plausibly a real lever, but on n = 20 the corr
  CI is wide; would expect this to replicate in a fresh run."*
- *"`min_mr` doesn't converge (CV = 0.92) and 38% of hops land at the lower
  bound. Either the feature is not strongly identified by the objective, or
  0 is the preferred value and the bound could be removed. The two readings
  predict different things if we widen the bound — recommend widening to
  test."*
- *"`se_span` shows no detectable correlation with the objective (Spearman
  = +0.04) and bounces across its range. On this sample we can't reject
  'feature is doing nothing,' but absence of evidence ≠ evidence of absence
  — consider whether the model code uses `se_span` somewhere that we're
  not measuring before dropping it."*

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

- *"Brier tracks across train and test (Spearman = +0.38, n = 20). On this
  sample size the CI is wide, but the relationship is positive and direct.
  Consistent with 'negative test delta is small-sample noise, not overfit'
  more than with 'we are overfitting train.'"*
- *"ATS performance shows no detectable train→test rank correlation
  (Spearman = +0.06). Either we are finding train ATS noise, OR the ATS
  signal is too low per-game to register on this test pool. Both readings
  imply 'don't trust train ATS as a selection criterion,' but they differ
  on whether the tuning has any underlying value."*

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

- *"Regression efficiency appears real and transfers: train Brier and train
  ats_be correlate +0.67, test Brier and test ats_be +0.56, and train Brier
  predicts test ats_be at +0.50 — consistent with shedding noise plays
  while preserving signal plays rather than collapsing into the market. On
  n = 20 these correlations are noisy; treat the direction as established,
  the magnitudes as approximate."*
- *"Brier and ats_be are uncorrelated on test (+0.04). On this sample we
  can't distinguish 'train-side efficiency is fitting noise' from 'test
  pool is too small to detect the transfer.' Either way, train ats_be is
  not a reliable selection criterion on this run."*

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

### Picking philosophy (applies to all stages)

These priorities sit above the stage-specific rankings below. When two
candidates score similarly on the stage-specific metric, the
philosophy decides.

1. **Train evidence outweighs test evidence.** Train pools are 10–25×
   larger than test pools (most NFL data lives there). A clear train
   winner with a slightly worse test number is more credible than a
   marginal test winner that was middling on train. Method 2's train↔test
   rank correlation tells you how much to discount test signal at all; on
   small samples test deltas are mostly noise.

2. **Heavily value low deviation from the current production config.**
   The deployed config has been validated in real conditions. A candidate
   that differs from production on 2–3 features by modest amounts is far
   less risky to ship than a candidate that differs on 7+ features by
   large amounts, even if the latter shows a marginal performance edge in
   training. Quantify deviation as Euclidean distance in normalized
   feature space (use the same normalization as Method 5). Use this as a
   tiebreaker AND as a hard penalty: a config that's "tied" with
   production but far from it in feature space is *worse* than one that's
   tied and close.

3. **Performance ties default to the closer-to-production candidate.**
   If two candidates land within sampling noise on the relevant train
   metric (and Method 2 says test isn't decisive), pick the one with
   smaller distance to production. Bias toward continuity unless there's
   real evidence the alternative is better.

4. **A bigger structural shift demands stronger evidence.** A candidate
   that represents a different feature-group profile from production
   (e.g., different `score_margin` regime, different
   `preseason_priors_share` valley) should not be picked on a small
   train edge alone. Require that the structural shift is *both*
   measurably better on train AND not contradicted on test. Without
   that, the production-adjacent config wins.

5. **Method 5 cluster density is a secondary signal, not a primary
   one.** Cross-run cluster robustness (Method 5 in group-score space)
   is real evidence the optimizer prefers a particular valley, but it
   doesn't override deviation-from-production. A robust cluster far
   from production is interesting but not picking-decisive unless it
   also delivers performance.

The ethos: **continuity is cheap, deviation is expensive, and
the burden of proof sits on the alternative**.

### Stage 1 (base model, unregressed Brier objective)

We want **the best market-unaware model**. Overfit avoidance is the binding
concern.

1. Take the top-K configs by train Brier (e.g., K = 10).
2. Compute deviation-from-production for each (normalized Euclidean in
   feature space).
3. Apply Method 5 to score robustness (group-score space preferred).
4. Tag each candidate's group profile (where it sits on
   `elo_intensity`, `preseason_priors_share`, etc. relative to
   production).
5. Apply the picking philosophy: prefer the candidate that scores well
   on train AND sits close to production AND doesn't represent a
   structural regime change without evidence on both train and test.
6. Sanity-check on the held-out test split: the picked config should at
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

## Calibration — how confident to be

The optimization problem is noisy and most analyses here run on small
samples. Be a grounded analyst, not a hyper-confident one. The user does
not want milquetoast "could be many things" hedging on every sentence, but
they also do not want confident causal claims that flip the moment a new
slice of evidence comes in.

**What the data actually is.**

- A typical run is ~20 random restarts. That is 20 *observations* of where
  the optimizer landed under different seeds, not 20 independent samples of
  "model quality." Two restarts with the same params are not independent
  trials of an i.i.d. experiment.
- The train pool is ~25 NFL seasons (~6500 games). Brier on that pool is
  reasonably stable; an observed delta of <100 brier between two
  configurations on this pool is inside random restart-to-restart variance.
- The test pool is 1–2 seasons (~270–570 games). Test brier carries
  enough variance that a 50–100 brier delta between configurations does
  not reliably distinguish them.
- NFL data is non-stationary. The optimizer is fitting joint behavior
  across seasons that differ in scoring environment, rule changes, kickoff
  rules, and market microstructure. A pattern that holds 2008–2023 can
  reverse in 2024–2025 for reasons unrelated to optimization quality.
- Spearman/Pearson correlations across 20 points have wide confidence
  intervals. A reported r = 0.4 on n = 20 has 95% CI roughly (−0.05, 0.72).
  Treat |r| < 0.3 on small samples as "no clear relationship" rather than
  "weak signal."

**Use this language ladder for findings.**

| What the data shows | How to say it |
|---|---|
| Direction consistent across multiple lenses (per-restart bests, head-to-head, train/test rank corr) AND magnitude exceeds restart-to-restart noise | "X improves Y by ~Z" — direct claim |
| Direction consistent but magnitude inside noise | "X looks slightly better on Y, but inside restart-to-restart variance" |
| Direction holds on train but not on test, or only on one lens | "On train we see X; on test the relationship weakens / reverses — could be overfit OR small-sample variance" |
| Conflicting lenses | "We see X by one cut and Y by another — these can't both be the right read; recommend more data before deciding" |
| Plausible mechanism for an observation | "One plausible mechanism is X. Verifiable by Y. Could equally be Z." |

**Causal claims need a verifier.** A statement like "the new translation is
overfitting because of per-season Platt drift" is a *hypothesis*, not a
finding. State it as "hypothesis: ..., would predict ..., test by ...".
Do not present it as the conclusion of the analysis.

**Sources of variance to consider before declaring a result:**

1. Random restart seed (how SLSQP started)
2. Train-set composition (which seasons; non-stationary)
3. Test-set size (1-2 seasons is noisy)
4. Translation-layer changes (rotate gradients; this conversation showed
   `margin_weight` sign flip with no actual change in NFL data)
5. Joint changes elsewhere in the pipeline since the prior run
6. The optimizer's tolerance/step parameters

If the result could plausibly be explained by any of (1)-(6) and the
sample doesn't rule it out, hedge the conclusion.

**One-eighty avoidance.** Before writing a confident statement, ask "what
single piece of evidence would flip my view?" If a single test-split column
or a single new lens could reverse the conclusion, the conclusion was too
strong. Lead with the observation, then the hypotheses, then the
recommendation framed as conditional on which hypothesis holds.

---

## Output expectations

The output is a **qualitative interpretation** of the run informed by the
quantitative analysis above — not a stat dump.

1. Open with **which stage** this run is and **what it was trying to do**
   (one sentence: "Stage-2 market-factor tune over {features}, objective
   `nfelo_brier_close`, train 2008-2023, test 2024-2025").
2. State the **headline finding** at a confidence level the data supports
   (one sentence: real progress / lucky once / regression / overfit / *or
   "results are mixed and small-sample"*). Calibrate per the ladder above.
3. Walk through the methods **interpretation first, metric as evidence**.
   Do not lead with raw numbers. Where a result depends on assumptions
   that could be wrong, say so inline — do not bury it as a footnote.
4. Flag any **known-tradeoff signals** that the run confirmed or contradicted.
5. **Separate observations from hypotheses from recommendations.** An
   observation is "metric X on lens Y has value Z." A hypothesis is "we
   think Z happens because W." A recommendation is "given X, do Y." Mark
   hypotheses as hypotheses and recommend a way to test them.
6. End with a **specific next action** — widen a bound, drop a feature,
   re-run with different test seasons, adopt a config, run a stage-1 retune
   before stage-2, etc. Tie it to a feature or a bound, not a vague gesture.
   If the data does not support a strong next action, say "wait for more
   restarts" rather than confecting one.

Total length: ~20-30 lines. The numbers in parentheses are evidence. A good
analysis demonstrates understanding of what each feature does in the model
and what the objective rewards, not just that the stats were computed —
and demonstrates honest calibration about what the data can and can't tell
you.
