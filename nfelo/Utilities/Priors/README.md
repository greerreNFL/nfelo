# Priors

Nfelo's offseason starting elo is a weighted blend of mean-reverted ending elo and external priors (DVOA, WT ratings, units preseason elo). Those inputs come from different models on different scales. This module converts each raw prior to nfelo elo before the blend.

Without that step, optimizer weights on `dvoa_weight`, `wt_ratings_weight`, and `units_weight` would partly be doing scale correction instead of signal mixing.

## Normalization

Each prior is z-scored against a fixed mean and a season-specific denominator, then mapped to elo:

```
prior_elo = 1505 + ((raw - μ) / σ_season) × prior_elo_scale
```

| Parameter | Where it lives | Role |
|-----------|----------------|------|
| `μ` | `config/{prior}.json` | Fixed center of the prior's raw distribution (0 for WT/DVOA, 1505 for units) |
| `σ_season` | `config/{prior}.json` → `normalization` | Walk-forward denominator for that season |
| `prior_elo_scale` | nfelo `config.json` | Elo points per one within-prior standard deviation after z-scoring |

`prior_to_elo()` applies this at runtime. It reads the precomputed `normalization` table — it does not recompute σ during a model update.

## Computing σ (offline)

`update_all_priors()` rebuilds the `normalization` tables from a team-season panel of raw values.

For each season `S`:

1. Take every `(team, season, raw)` row with `season < S`
2. Weight each row by how far its season is from `S` (half-life decay; uniform if `half_life` is `null`)
3. Compute weighted standard deviation around fixed `μ`
4. Store result as `σ_S`

No data from season `S` or later enters `σ_S`. That is the leak-free constraint.

After a data refresh or half-life change, rerun:

```python
from nfelo.Utilities.Priors import update_all_priors

update_all_priors()
```

Half-life is set manually in each prior's JSON. This module does not tune it.

## Half-life

Half-life only affects how past seasons are weighted when building σ. It does not change raw prior values or within-season rankings. It changes how wide or narrow the normalized prior looks in elo space relative to history.

Current settings:

**WT — `null` (uniform history)**

WT's cross-sectional spread has been stable relative to DVOA and units. There is no need to down-weight older seasons when computing σ. Uniform weighting gives a well-sampled denominator for a prior with a long track record.

**Units — `null` (uniform history)**

Units preseason elo is already on an elo scale (`μ = 1505`). Raw dispersion has drifted wider over time; σ computed from full history absorbs that drift in the denominator. Recency weighting is unnecessary for mapping units into nfelo's blend — the prior's job is level and rank, not matching year-to-year changes in raw spread.

**DVOA — `3`**

DVOA projections vary more in cross-sectional dispersion across eras (methodology changes, compression/expansion in projected spreads). A short half-life lets σ track recent history so normalized DVOA stays on comparable footing with WT and units when blended. WT and units do not need the same responsiveness.

## Data

Raw values loaded for σ updates (`load_panel.py`):

| Prior | Column | Source |
|-------|--------|--------|
| wt | `wt_rating` | `dcm.load(['wt_ratings'])` |
| dvoa | `projected_dvoa` | `Intermediate Data/dvoa_projections.csv` |
| units | `units_preseason_elo` | `dcm.load(['nfelounits_elo'])`, first week per team-season |

Game-level units elo is aggregated to team-season preseason values in `DataLoader.add_units_elo()` using the same first-week logic.

## Runtime path

Week 1 of each season, `Nfelo.py` passes raw prior values into `offseason_regression()`, which calls `prior_to_elo()` per prior and blends with mean-reverted ending elo.

Missing or invalid inputs do not raise. If a season has no σ in the normalization table, or the raw value / σ is null or non-positive, `prior_to_elo()` returns the team's mean-reverted elo — that prior's weight is effectively reallocated to the internal prior for that team. This keeps week 1 projections publishable when upstream data is incomplete, but it means a broken merge degrades silently; run `update_all_priors()` after extending the panel to a new season so lookups don't fall back.

Note on caching: prior configs are read from disk once at module import. A process that calls `update_all_priors()` will keep using the old σ tables until it is restarted — run updates and model runs in separate invocations.
