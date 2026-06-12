'''
Series-level helpers for assembling market implied home win probabilities
in DataLoader.format_market_data.

The nfelotranslation Translator is a scalar per-game API, so spread<->WP
conversion loops over a Series while reusing the per-season fit via
.update() within a season. Spread/ML blending is logit-space numpy math.

Sign convention:
    nfelo uses sportsbook spreads (negative = home favored).
    nfelotranslation uses positive = home favored.
    Spread inputs are negated on the way IN and spread outputs are negated
    on the way OUT. Win probability is side-agnostic and not transformed.
'''
import pandas as pd
import numpy
from scipy.special import expit
from scipy.special import logit

from nfelotranslation import Translator

def _translate_series(values:pd.Series, seasons:pd.Series, input_type:str, in_xform:callable, out_xform:callable):
    '''
    One Translator reused via .update() within a season;
    rebuilt when the season changes. Pre-2007 seasons clamp to 2007.
    NaN rows pass through as NaN in the output.

    Parameters:
    * values (series):  per-game numeric inputs in nfelo convention
    * seasons (series): season corresponding to each value (index aligned)
    * input_type (str): nfelotranslation input_type ('win_prob', 'spread', ...)
    * in_xform (callable):  value transform from nfelo -> nfelotranslation convention
    * out_xform (callable): translator -> output value in nfelo convention

    Returns:
    * series of translated values, index aligned with values
    '''
    ## clamp to the earliest season of the nfelotranslation package ##
    _EARLIEST_SEASON = 2007
    ## convert to numpy arrays for speed ##
    season_arr = seasons.to_numpy()
    value_arr = values.to_numpy(dtype=float)
    out = numpy.full(len(values), numpy.nan)
    ## initialize translator and season ##
    translator = None
    translator_season = None
    for i in range(len(values)):
        value = value_arr[i]
        season = season_arr[i]
        if numpy.isnan(value) or pd.isna(season):
            continue
        nt_value = in_xform(value)
        season_clamped = max(int(season), _EARLIEST_SEASON)
        if translator is None or translator_season != season_clamped:
            translator = Translator(nt_value, input_type, season=season_clamped, side='home')
            translator_season = season_clamped
        else:
            translator.update(nt_value, input_type)
        out[i] = out_xform(translator)
    return pd.Series(out, index=values.index)

def market_spread_to_win_prob_series(spreads:pd.Series, seasons:pd.Series):
    '''
    Translates a series of market spreads (sportsbook convention) to
    home win probabilities via Translator(input_type='spread').

    Parameters:
    * spreads (series): home-perspective market spreads (negative = home favored)
    * seasons (series): season corresponding to each spread (index aligned)

    Returns:
    * win_probs (series): home win probabilities from unified SpreadMapper (index aligned)
    '''
    return _translate_series(
        spreads, seasons,
        input_type='spread',
        in_xform=lambda v: -v,                ## nfelo sportsbook -> nfelotranslation positive=home favored ##
        out_xform=lambda t: t.win_prob,       ## home win probability from unified SpreadMapper ##
    )

def blend_spread_ml_win_prob_series(
    spread_wp:pd.Series,
    ml_wp:pd.Series,
    spread_weight:float=0.7,
):
    '''
    Blend spread-implied and ML-implied home win probabilities in logit space.

    When both are present:
        expit(spread_weight * logit(spread_wp) + (1 - spread_weight) * logit(ml_wp))
    When only one is present: 100% of the available signal.

    Parameters:
    * spread_wp (series): spread-implied home win probability
    * ml_wp (series): hold-adjusted ML-implied home win probability
    * spread_weight (float): logit-space weight on spread when both exist (default 0.7)

    Returns:
    * blended_wp (series): combined home win probability (index aligned)
    '''
    _WP_CLAMP_LO = 0.001
    _WP_CLAMP_HI = 0.999
    has_spread = spread_wp.notna().to_numpy()
    has_ml = ml_wp.notna().to_numpy()
    spread_arr = spread_wp.to_numpy(dtype=float)
    ml_arr = ml_wp.to_numpy(dtype=float)
    out = numpy.full(len(spread_wp), numpy.nan)
    both = has_spread & has_ml
    spread_only = has_spread & ~has_ml
    ml_only = ~has_spread & has_ml
    spread_clamped = numpy.clip(spread_arr[both], _WP_CLAMP_LO, _WP_CLAMP_HI)
    ml_clamped = numpy.clip(ml_arr[both], _WP_CLAMP_LO, _WP_CLAMP_HI)
    out[both] = expit(
        spread_weight * logit(spread_clamped) +
        (1.0 - spread_weight) * logit(ml_clamped)
    )
    out[spread_only] = spread_arr[spread_only]
    out[ml_only] = ml_arr[ml_only]
    return pd.Series(out, index=spread_wp.index)

def win_prob_to_model_spread_series(win_probs:pd.Series, seasons:pd.Series):
    '''
    Translates a series of home win probabilities to implied posted model
    spreads (sportsbook convention) via Translator(input_type='win_prob').

    Parameters:
    * win_probs (series): home win probabilities
    * seasons (series): season corresponding to each value (index aligned)

    Returns:
    * spreads (series): implied posted home spreads (sportsbook, 0.5-grained)
    '''
    return _translate_series(
        win_probs, seasons,
        input_type='win_prob',
        in_xform=lambda v: v,                 ## win prob is side-agnostic; no transform ##
        out_xform=lambda t: -t.spread.posted, ## nfelotranslation positive=home favored -> nfelo sportsbook ##
    )
