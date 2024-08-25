import pandas as pd
import numpy
from .base import is_series

def grade_bet_vector(
      model_line:pd.Series,
      market_line:pd.Series,
      result:pd.Series,
      home_ev:pd.Series,
      away_ev:pd.Series,
      be_only:bool
) -> pd.Series:
    '''
    Grades a series of model and market lines as ats bets. If
    break-even only is flagged, only break=even bets will be grade,
    with all else returning nan.

    If no EV is provided, but be_only is set to true, then break even is
    assumed to be any spread > 1.5 points from the market

    Parameters:
    * model_line: model's home spread
    * market_line: market's home spread
    * result: final home margin
    * home_ev: expected value of a bet on the home team
    * away_ev: expected value of a bet on the away team
    * be_only: only include bets with break even or better EV
    '''
    ## establish EV for null ##
    ## spread delta ##
    home_spread_delta = model_line-market_line
    away_spread_delta = market_line-model_line
    ## if home ev is missing, set home ev to based on spread delta ##
    home_ev = numpy.where(
        pd.isnull(home_ev),
        numpy.where(
          home_spread_delta > 1.5,
          0.05,
          -0.05
        ),
        home_ev
    )
    away_ev = numpy.where(
        pd.isnull(away_ev),
        numpy.where(
          away_spread_delta > 1.5,
          0.05,
          -0.05
        ),
        away_ev
    )
    ## now determine plays ##
    ## if be_only is not flagged, all plays are valid, otherwise
    ## breakeven or better only ##
    if be_only:
        count_as_play = (home_ev > 0) | (away_ev > 0)
    else:
        count_as_play = ~pd.isnull(home_ev)
    ## determine wins and losses ##
    bet_outcome = numpy.where(
        ## pushes and non-plays ##
        (model_line == market_line) | (market_line + result == 0),
        numpy.nan,
        numpy.where(
            ## home bet ##
            ## note: flip sign here to denote MoV vs spread
            ##       More intuitive to think of a greater model line as 
            ##       an expectation that home is better rel to market
            -model_line > -market_line,
            ## home result ##
            numpy.where(
                result + market_line > 0,
                1,
                0
            ),
            ## away bet ##
            numpy.where(
                -model_line < -market_line,
                ## away result ##
                numpy.where(
                    result + market_line < 0,
                    1,
                    0
                ),
                ## should not be reachable ##
                numpy.nan
            )
        )
    )
    ## put it all together ##
    graded = numpy.where(
        count_as_play,
        bet_outcome,
        numpy.nan
    )
    ## return ##
    return graded
