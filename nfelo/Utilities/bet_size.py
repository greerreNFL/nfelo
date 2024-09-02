import pandas as pd
import numpy
import numpy.typing as npt

def kelly_bet_size(
    win_prob:npt.ArrayLike,
    odds:(None or npt.ArrayLike)=None
) -> npt.ArrayLike:
    '''
    Calculates the kelly criterion betsize for an array of win probs. Currently defaults to
    to -110 odds for all all calcs, but this will change to be adjustable in the future
    '''
    if odds is None:
        odds = 1/1.1
    return win_prob - ((1-win_prob) / odds)

def bet_size(
    home_cover_prob:pd.Series,
    home_loss_prob:pd.Series,
    total_risk_target:float=50000,
    max_bet_pct:float=0.024,
    min_bet_pct:float=0.005
) -> pd.Series:
    '''
    A modified approach to discounting the Kelly Criterion
    that seeks to maintain the ethos of scalling bet size with
    edge while controlling the total expected amount risked on a season

    The general structure is to first set the full kelly to logrithmic
    curve rather than an exponential one. Greater edges always receiver
    a higher size, but the difference between large and small bets is dramatically
    reduced

    This new percent is then multiplied by a "max bet" size rather than
    
    Parameters:
    * home_cover_prob:
    * home_loss_prob:
    * total_risk_target: Total amount to be bet on the season
    * max_bet_pct: Max bet as a percent of total risk
    * min_bet_pct: Min bet as a percent of total risk

    Returns:
    * bet_size: A rounded amount to risk
    '''
    ## get normalized win/loss prob, controlling for home / away
    ## first get away probs ##
    acp = 1 - home_cover_prob
    alp = 1 - home_loss_prob
    ## normalize ##
    acp_norm = acp / (acp+alp)
    hcp_norm = home_cover_prob / (home_cover_prob + home_loss_prob)
    ## max prob ##
    prob = numpy.maximum(hcp_norm,acp_norm)
    ## translate to kelly ##
    kelly = kelly_bet_size(prob)
    ## scale the kelly bet size ##
    scaled_kelly = numpy.log(1 + kelly*100) / numpy.log(101)
    ## multiply by the max bt size ##
    bet_size = scaled_kelly * max_bet_pct * total_risk_target
    ## round ##
    bet_size_round = numpy.round(bet_size / 50) * 50
    ## remove bets under minimum ##
    bet_size_final = numpy.where(
        bet_size_round > min_bet_pct * total_risk_target,
        bet_size_round,
        numpy.nan
    )
    ## return ##
    return bet_size_final
