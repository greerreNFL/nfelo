import numpy
import pandas as pd
from typing import Dict, Optional
from ...constants import DEFAULT_MIN_HISTORY_SEASONS


def _season_weight(season_t: int, season_s: int, half_life: Optional[float]) -> float:
    if half_life is None:
        return 1.0
    return 0.5 ** ((season_s - season_t) / half_life)


def _weighted_sigma(values, weights, mu: float) -> float:
    v = numpy.asarray(values, dtype=float)
    w = numpy.asarray(weights, dtype=float)
    mask = numpy.isfinite(v) & numpy.isfinite(w) & (w > 0)
    v = v[mask]
    w = w[mask]
    if len(v) == 0:
        return numpy.nan
    var = numpy.average((v - mu) ** 2, weights=w)
    return float(numpy.sqrt(max(var, 0.0)))


def compute_normalization(
    panel: pd.DataFrame,
    value_col: str,
    mu: float,
    half_life: Optional[float] = None,
    min_history_seasons: int = DEFAULT_MIN_HISTORY_SEASONS,
) -> Dict[int, float]:
    '''
    Compute leak-free season -> sigma tables from a team-season panel.

    Parameters:
    * panel (pd.DataFrame): team-season panel with raw prior values
    * value_col (str): column name for the prior raw value
    * mu (float): fixed prior mean
    * half_life (float): optional decay for weighting past seasons
    * min_history_seasons (int): minimum prior seasons required before emitting sigma

    Returns:
    * normalization (dict[int, float]): season -> sigma
    '''
    if value_col not in panel.columns:
        raise KeyError('PRIOR ERROR: Panel missing required column: {0}'.format(value_col))
    clean = panel[['team', 'season', value_col]].dropna(subset=[value_col]).copy()
    normalization = {}
    ## walk forward by season ##
    for season_s in sorted(clean['season'].unique()):
        past = clean[clean['season'] < season_s].copy()
        if past.empty:
            continue
        if past['season'].nunique() < min_history_seasons:
            continue
        ## weight past seasons ##
        past['weight'] = past['season'].map(
            lambda t: _season_weight(int(t), int(season_s), half_life)
        )
        sigma = _weighted_sigma(past[value_col], past['weight'], mu)
        if not numpy.isfinite(sigma) or sigma <= 0:
            continue
        normalization[int(season_s)] = sigma
    return normalization
