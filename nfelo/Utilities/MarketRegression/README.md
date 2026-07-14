# Market Regression

Markets hold valuable information that a prediction model can benefit from. Shading or moving a models projection towards the market is referred to as "regression".

Though regression improves overall prediction accuracy, it also makes the model track the market more closely, which can mute its opinionatedness and edge. The accuracy<>alpha trade-off is what market regression is meant to balance. It tries to preserve opinionatedness in contexts where it is right, and squash opinionatedness in contexts where it is wrong.

At the highest level, this means:
- Collapsing to the market when the model's delta is small enough to be within a margin of error.
- Reduce extreme opinions to the point where enough delta is preserved to see value, but no more.
- Incorporate market movement between open and close, respecting that closing lines are more accurate than opening lines.

## Design

**Target.** Regression is toward the existing blended market elo dif (`market_elo_dif_open` / `market_elo_dif_close` from the DataLoader). The blend itself is unchanged.

**Open — logistic S-curve.** Disagreement `d = |base - market_open|` maps to a regression factor via a standard logistic. Asymptotes are 1.0 (full regression) and `min_mr`. The logit center is shifted so the factor equals 0.5 exactly at `mr_mid`. `mr_steep` is the logistic steepness in 1/elo units.

**Close — CPR.** The close does not get an independent S-curve pass. Its factor is the open's effective factor (`PR`), rescaled by the movement ratio `r = CD / OD` through an odds transform:

```
CPR = PR * r^mr_close_exp / (PR * r^mr_close_exp + (1 - PR))
```

- `r = 1` (no movement): close treated exactly like open
- `r > 1` (moved away): cushion shrinks; large adverse moves → near-full regression
- `r < 1` (moved toward us): conviction restored toward the raw number
- No opening opinion (`OD ≈ 0` or `PR = 1`): CPR = 1 — movement can never create an opinion

**Anchoring.** Both legs start from the raw `base` dif and move toward the market. The close opinion cannot be more extreme than the raw number relative to the close market. A residual cap (`mr_cap_elo`) mutes extreme posted opinions for brier without changing play decisions at the defaults.

## Formulas

Logistic S-curve factor (`s_curve.py`):

```
s* = (0.5 - min_mr) / (1 - min_mr)
x0 = mr_mid - log(1/s* - 1) / mr_steep

unit   = 1 / (1 + exp(mr_steep * (d - x0)))
factor = min_mr + (1 - min_mr) * unit
```

Open / close regression (same shape; factor is logistic on open, CPR on close):

```
reg = base + factor * (market - base)
reg = market + clip(reg - market, -mr_cap_elo, +mr_cap_elo)
```

Effective factor returned to the model (and consumed as `PR` by the close leg):

```
PR = (reg - base) / (market - base)   if |base - market| > eps else 1.0
```

## Config

| Key | Default | Role |
|-----|---------|------|
| `mr_mid` | 50.0 | Disagreement (elo) where the open factor equals 0.5 |
| `mr_steep` | 0.45 | Logistic steepness in 1/elo units |
| `min_mr` | 0.4345 | Asymptotic floor for the open S-curve |
| `mr_cap_elo` | 60.0 | Max absolute residual vs market after regression |
| `mr_close_exp` | 2.0 | Sensitivity of CPR to the movement ratio |

`se_span` remains in nfelo config for rolling SE tracking elsewhere; it is no longer an MR input.

## Runtime path

In `Nfelo.project_game`:

1. Build `base` = raw elo dif after HFA / QB / bye / playoff boost
2. `regress_markets(base, market_elo_dif_open, market_elo_dif_close, ...)` → open/close difs and factors
3. Translate both legs with `elo_to_prob` + Translator as before

`regress_markets` owns sequencing and missing-market handling: normal path is open then CPR close; if open is missing but close exists, close gets an open-style pass on the close market (same as zero movement); missing legs return null.

## Modules

| File | Responsibility |
|------|----------------|
| `s_curve.py` | `s_curve_factor` |
| `regress_open.py` | `regress_open` |
| `regress_close.py` | `close_pr`, `regress_close` |
| `regress_markets.py` | `regress_markets` (open→close orchestration) |
