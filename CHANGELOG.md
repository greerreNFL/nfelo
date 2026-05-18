# Changelog

All notable changes to this project will be documented in this file.

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
