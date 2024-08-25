import pandas as pd
import numpy
import pathlib
import math

from .base import is_series

## file location ##
file_loc = pathlib.Path(__file__).parent.resolve()

## load the spread <> prob maps as data frames ##
spread_to_prob = pd.read_csv(
    '{0}/datasets/spread_probability_translation.csv'.format(file_loc)
)
prob_to_spread = pd.read_csv(
    '{0}/datasets/probability_spread_multiples.csv'.format(file_loc)
)

## round values so floats can be keys in the dictionary lookup ##
spread_to_prob['spread'] = spread_to_prob['spread'].round(3)
spread_to_prob['implied_win_prob'] = spread_to_prob['implied_win_prob'].round(3)
prob_to_spread['win_prob'] = prob_to_spread['win_prob'].round(3)
prob_to_spread['implied_spread'] = prob_to_spread['implied_spread'].round(3)

## determine range bounds so lookups dont fail ##
spread_to_prob_max = spread_to_prob['spread'].max()
spread_to_prob_min = spread_to_prob['spread'].min()

prob_to_spread_max = prob_to_spread['win_prob'].max()
prob_to_spread_min = prob_to_spread['win_prob'].min()

## translate into dictionaries for mapping ##
spread_to_prob_dict = dict(zip(spread_to_prob['spread'],spread_to_prob['implied_win_prob']))
prob_to_spread_dict = dict(zip(prob_to_spread['win_prob'],prob_to_spread['implied_spread']))
## create a string key version to avoid floating point issues ##
spread_to_prob_dict_str = {f'{k:.1f}': v for k, v in spread_to_prob_dict.items()}
prob_to_spread_dict_str = {f'{k:.3f}': v for k, v in prob_to_spread_dict.items()}

## functions to perform translates ##
## SPREAD TO PROBABILITY ##
def spread_to_probability_vector(spreads:pd.Series):
    '''
    Translates a series of spreads to probabilites

    Parameters:
    * spreads (series): game spreads

    Returns:
    * probabilities (series): implied win probabilities
    '''
    return spreads.round(3).clip(
        upper=spread_to_prob_max,
        lower=spread_to_prob_min
    ).map(spread_to_prob_dict)

def spread_to_probability_float(spread:float):
    '''
    Translates a spread to a probability

    Parameters:
    * spread (float): game spread

    Returns:
    * probability (float): implied win probability
    '''
    ## ensure spread is an inbounds key ##
    spread_ = round(
        min(spread_to_prob_min, max(spread_to_prob_max, spread)),
        3
    )
    return spread_to_prob_dict[spread_]

def spread_to_probability(spread:(pd.Series or float)):
    '''
    Translates a spread or series of spreads to implied win probability

    Paramaters:
    * spread (series, float): game spread

    Returns:
    * probability (series, float): implied win probabilities
    '''
    if is_series(spread):
        return spread_to_probability_vector(spread)
    else:
        return spread_to_probability_float(spread)
    

## PROBABILITY TO SPREAD ##
def probability_to_spread_vector(probabilities:pd.Series):
    '''
    Translates a series of probabilities to spreads

    Parameters:
    * probability (series): win probability

    Returns:
    * spreads (series): implied spreads
    '''
    ## use strings to do perform the lookup ##
    return pd.Series(numpy.char.mod('%.3f', probabilities.round(3).clip(
        upper=prob_to_spread_max,
        lower=prob_to_spread_min
    )), index=probabilities.index).map(prob_to_spread_dict_str)

def probability_to_spread_float(probability:float):
    '''
    Translates a win probability to its implied spread

    Parameters:
    * probability (float): win probability

    Returns:
    * spread (float): implied win spread
    '''
    ## ensure spread is an inbounds key ##
    prob_ = round(
        max(prob_to_spread_min, min(prob_to_spread_max, probability)),
        3
    )
    return prob_to_spread_dict[prob_]

def probability_to_spread(probability:(pd.Series or float)):
    '''
    Translates a spread or series of win probabilities to their implied spreads

    Paramaters:
    * probability (series, float): win probability

    Returns:
    * spread (series, float): implied spread
    '''
    if is_series(probability):
        return probability_to_spread_vector(probability)
    else:
        return probability_to_spread_float(probability)

def elo_to_prob(elo_dif:(int or float), z:(int or float)=400):
    '''
    Converts and elo difference to a win probability

    Parameters:
    * elo_dif (int or float): elo difference between two teams
    * z (int or float): config param that determines confidence

    Returns:
    * win_prob (float): the win probability implied by the elo dif
    '''
    ## handle potential div 0 ##
    if z <=0:
        raise Exception('NFELO CONFIG ERROR: Z must be greater than 0')
    ## return ##
    return 1 / (
        math.pow(
            10,
            (-elo_dif / z)
        ) +
        1
    )

def prob_to_elo(win_prob:(float), z:(int or float)=400):
    '''
    Converts a win probability to an elo difference. This is
    the inverse of elo_to_prob()

    Parameters:
    * win_prob (float): win probability of one team over another
    * z (int or float): config param that determines confidence

    Returns:
    * elo_dif (float): implied elo dif between the teams
    '''
    ## handle potential div 0 ##
    if z <=0:
        raise Exception('NFELO CONFIG ERROR: Z must be greater than 0')
    ## return the dif ##
    return (
        (-1 * z) *
        numpy.log10(
            (1/win_prob) -
            1
        )
    )