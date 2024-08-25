import pandas as pd
import numpy

from .base import is_series

## AMERICAN ODDS TO PROBABILITY ##
def american_to_prob_int(num:int):
    '''
    Convert and int american prob to a probability
    '''
    if num < 100:
        return (-1 * num) / (100 - num)
    else:
        return 100 / (100 + num)

def american_to_prob_vector(series:pd.Series):
    '''
    Convert a series of american odds into a series of probabilties
    '''
    return numpy.where(
        series < 100,
        (-1 * series) / (100 - series),
        100 / (100 + series)
    )

def american_to_prob(odds:(pd.Series or int)):
    '''
    Calculates the implied probability of american odds

    Parameters:
    * odds (series, or int): American odds

    Returns:
    * probability (series, or float): Implied probability
    '''
    if is_series(odds):
        return american_to_prob_vector(odds)
    else:
        return american_to_prob_int(odds)

def american_to_hold_adj_prob(home_odds:(pd.Series or int), away_odds:(pd.Series or int)):
    '''
    Calculates the hold adjusted probability from american odds

    Parameters:
    * home_odds (series, or int): Home team's american odds
    * away_odds (series, or int): Away team's american odds

    Returns:
    * home_probability (series, or float): Implied probability of the home team
    * away_probability (series, or float): Implied probability of the away team
    * hold(series, or float): Hold
    '''
    ## get implied odds individually
    home_prob = american_to_prob(home_odds)
    away_prob = american_to_prob(away_odds)
    combo_prob = home_prob + away_prob
    return (
        home_prob / combo_prob,
        away_prob / combo_prob,
        combo_prob - 1
    )


## AMERICAN TO PRICE ##
def american_to_price_int(odds:int):
    '''
    Convert an american odd into a cost basis
    '''
    if odds < 0:
        return abs(odds)
    else:
        return 100 / (odds / 100)

def american_to_price_vector(odds:pd.Series):
    '''
    Convert a series of american odds into a series of basis
    '''
    return numpy.where(
        odds < 0,
        numpy.absolute(odds),
        100 / (odds / 100)
    )

def american_to_price(odds:(pd.Series or int)):
    '''
    Calculates the amount that would be lost in a bet to win $100 provided odds

    Parameters:
    * odds (int, series): American odds

    Returns:
    * cost (float, series): Amount risked in a bet to win $100
    '''
    if is_series(odds):
        return american_to_price_vector(odds)
    else:
        return american_to_price_int(odds)

## SPREAD TO PROBABILITY ##
def spread_to_prob_elo(spread:(pd.Series or float)):
    '''
    Converts a spread line into a win probability using the 538s Elo
    methodology

    Parameters:
    * spread (series, float): the spread of the game

    Returns:
    * prob (series, float): the win probability of the team
    '''
    dif = spread * 25
    ## convert to prob ##
    prob = (
        1 /
        (
            10 ** (-dif / 400) +
            1
        )
    )
    return prob


## PROBABILITY TO ELO ##
def prob_to_elo(prob:(pd.Series or float), z:int=400):
    '''
    Converts a win probability into an Elo difference
    
    Parameters:
    * prob (series, or float): Win probability
    * z (int) - optional: Elo scaling factor

    Returns:
    * elo_dif (series, or float): The difference in Elo implied by the prob
    '''
    return numpy.log10(
        (1 / prob) - 1
    ) * -z

