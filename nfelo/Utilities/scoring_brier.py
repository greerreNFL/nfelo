import pandas as pd
import numpy
from .base import is_series


def brier_score_vector(
    probability:pd.Series, margin:pd.Series
) -> pd.Series:
    '''
    Calculates brier score for a series
    '''
    return numpy.where(
        margin > 0,
        25 - (((probability*100)-100)**2)/100,
        25 - ((((1- probability)*100)-100)**2)/100
    )

def brier_score_number(
    probability:(float), margin:(float)
) -> float:
    '''
    Calcualtes brier score for a single record
    '''
    if margin > 0:
        return 25 - (((probability*100)-100)**2)/100,
    else:
        return 25 - ((((1- probability)*100)-100)**2)/100


def brier_score(
    probability:(float or pd.Series),
    margin:(float or pd.Series)
):
    '''
    Calculates a modified Brier score using the approach
    created in the original 538 model

    Parameters:
    * probability (float/series): estiamted win probability
    * margin (float/series): result of the game

    Returns:
    * brier (float/series): scored result
    '''
    if is_series(probability):
        return brier_score_vector(probability, margin)
    else:
        return brier_score_number(probability, margin)
