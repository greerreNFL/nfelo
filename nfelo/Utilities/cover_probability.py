import pandas as pd
import numpy

import pathlib

## load distributions ##
dist_df = pd.read_csv(
    '{0}/datasets/margin_distributions.csv'.format(pathlib.Path(__file__).parent.resolve()),
    index_col=0
)

def calc_cover_probs(proj_spread:float, market_spread:float):
    '''
    Uses the margin distribution model to calculate loss, push and cover
    probabilities for a model spread given a market spread

    Parameters:
    * proj_spread: the projected spread
    * market_spread: the market's spread

    Returns
    * loss, push, cover: a tuple of probabilities

    Returns a loss probability, push probability, and cover probability
    '''
    ## flip the sign of the spreads to make them equivalent to
    ## the representation of the dist_df
    proj_spread = -1 * round(proj_spread, 1)
    market_spread = -1 * round(market_spread, 1)
    ## get the probability distribution for the projected model spread ##
    temp = dist_df[
        dist_df['spread_line'] == proj_spread
    ].copy()
    ## calc the probs ##
    loss = numpy.where(
        temp['result'] < market_spread,
        temp['normalized_modeled_prob'],
        0
    ).sum()
    push = numpy.where(
        temp['result'] == market_spread,
        temp['normalized_modeled_prob'],
        0
    ).sum()
    cover = numpy.where(
        temp['result'] > market_spread,
        temp['normalized_modeled_prob'],
        0
    ).sum()
    ## return ##
    return loss, push, cover