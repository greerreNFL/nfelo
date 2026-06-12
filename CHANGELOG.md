# Changelog

All notable changes to this project will be documented in this file.

## [4.1.0] - 2026-06-12

### Changed
- **Replaced CSV spread/probability lookup tables with
  `nfelotranslation` 0.2.0 `Translator`.** `Nfelo.project_game` now
  maps win probabilities to spreads and derives cover/push/loss
  probabilities.
- **`DataLoader` market implied win probabilities** now use
  `nfelotranslation` spread→WP conversion (`Data/Helpers/market_wp.py`)
  instead of `spread_to_probability` CSV lookup. Spread and ML implied
  probabilities are combined via a **logit blend (70% spread / 30% ML)**
  when both are available, rather than spread-only with ML fallback.
- **`calc_clv`** now takes a `season` argument.
- **`spread_translation.py`** trimmed to `elo_to_prob` / `prob_to_elo`
  only. Removed `probability_to_spread`, `spread_to_probability`, and
  the associated lookup CSVs.
- **Optimizer output schema standardized** via `RecordSchema.py`.
  Train CSV rows now carry all model metrics in canonical
  `{metric}_{model}` columns (fixed column order across objectives).
  Test rows use `test_`-prefixed columns joinable on `run_id`.
  `objective_model` and `objective_metric` columns added for clarity.
- **`_benchmarks.csv` snapshot** now writes scalar columns only (fixes
  a bug where a `home_line` Series was written into benchmark rows).
- **`ANALYSIS_PLAYBOOK.md`** updated for the new optimizer output
  schema and benchmark semantics.

### Added
- `nfelotranslation==0.2.0` dependency (`requirements.txt`).
- `Data/Helpers/market_wp.py` — series-level spread↔WP helpers and
  spread/ML logit blending for `DataLoader`.
- `Optimizer/Primitives/RecordSchema.py` — canonical models, metrics,
  features, and `extract_performance()` for optimizer CSVs.
- Per-eval runtime logging to `{opti_tag}-{date}_runtime.csv`
  (`eval_seconds`, hop, eval number, objective value).

### Removed
- `Utilities/cover_probability.py` and lookup datasets
  (`margin_distributions.csv`, `probability_spread_multiples.csv`,
  `spread_probability_translation.csv`).

## [4.0.2] - 2026-05-31

### Fixed
- **SE was computed against the wrong sign.** `Nfelo.process_game` was using
  `(margin - line)²` for `se_market` and `se_model`. With the canonical
  negative-home-favored convention, the expected home margin is `-line`, so
  the correct formula is `(margin + line)²`. The bug roughly doubled the
  per-team rolling SE that feeds `rmse_adj`, so the market-regression
  utility was reacting to systematically-inflated error magnitudes.
- **`rmse_adj` was effectively a no-op.** `regress_to_market` was passing
  `market_line, market_line` to `rmse_adj` instead of `model_line,
  market_line`. Since `rmse_adj` only activates when
  `|model_line − market_line| > 1`, that condition was never true and the
  function always returned 1, meaning the `rmse_base` config parameter had
  no effect on regression. Fixed.
- `Nfelo.__init__` was reading `config['begining_elo']` (typo). Renamed
  both sides to `beginning_elo`.
- `Model.Nfelo.process_game` was writing the away team's opponent to the
  home team's `current_elos` entry. Field is currently unread; corrects
  data for future `elo_records` consumers.

### Changed
- **Decomposed `NfeloOptimizer` into primitives and switched the
  multi-start strategy from basin-hopping to random starts.** New
  `Optimizer/Primitives/` holds `NfeloOptimizerBase` (one SLSQP local
  optimization, saves on new best) and `RandomStarts` (N independent
  restarts from uniform-random points). Random starts are a better fit
  than basin-hopping for this objective surface: each hop is independent
  (no chain state, no perturbation, no Metropolis acceptance), making the
  search faster, easier to reason about, and parallelizable in the future
  at the cost of finding a global optimum (which is likely overfit).
  Kwarg `basin_hop` renamed to `random_starts` on `NfeloOptimizer` and the
  `Development.optimization` helpers.
- Retuned the market-regression params in `config.json` (`nfelo_v4.02`).
  The retune was necessary because the two bug fixes above (SE sign,
  `rmse_adj` no-op) changed what the regression utility sees and how it
  responds. Updated: `se_span`, `rmse_base`, `spread_delta_base`,
  `long_line_inflator`, `hook_certainty`, `min_mr`.

### Added
- Train/test split support: `test_seasons` kwarg on `NfeloOptimizer` and
  `season_filter` on `NfeloGrader`. Test split is graded on every new best,
  giving generalization signal alongside the train objective.
- Per-run side-tables in `Optimizer/results/`: `_test.csv` (test metrics
  per `run_id`, joinable to the train CSV) and `_benchmarks.csv` (market
  and market_open baselines per split, snapshotted once).
- `Optimizer/ANALYSIS_PLAYBOOK.md` — methodology for analyzing runs.

## [4.0.1] - 2026-05-17

### Removed
- The `Analytics` subpackage (`NfeloAnalytics`) and its public re-export from `nfelo.__init__`. It's output ( `team_file.csv` and `most_recent_team_file.csv`) are no longer used by any downstream consumers.
- Stale entries from `config.json` (`secondary_output_path`, `data_pulls`, `formatting`, `models.wepa`, `models.spreads`, `models.wt_ratings`, and unused `models.nfelo.*` file-path / metadata keys).

### Fixed
- `Utilities.market_regression.hook_adj` no longer crashes when `market_line`
  is `NaN`. Returns a neutral hook factor of `1` in that case so offseason
  projections (Week 1 of an upcoming season, before books post lines) complete
  without raising `ValueError: cannot convert float NaN to integer`.

### Added
- `nfelo.__version__` attribute, sourced from the inner `nfelo/__init__.py`.
- This `CHANGELOG.md`.

## [4.00]

### Changed
- Complete rebuild of the nfelo package. Version history will be regularly
  maintained going forward.
