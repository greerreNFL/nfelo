import numpy
import math


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