import pandas as pd
import math

def calc_shift(
    margin_measure:float, model_line:float, market_line:float,
    k:(float or int), b:(float or int), market_resist_factor:(float or int),
    is_home:bool
) -> float:
    '''
    Calculates the amount of elo to shift the team

    Parameters:
    * margin_measure (float): the value used to measure the game outcome
    * model_line (float): the line generated by the model
    * market_line (float): the closing market line
    * k (float or in): model param that effects degree of shift
    * b (float or int): model param that effects certainty in outcome
    * market_resist_factor (float or int): determins how much to adjust back to
    the market when the model is wrong
    * is_home (bool): whether the home team is being adjusted (as opposed to away team)

    Returns:
    * elo_shift (float): how much the teams elo should be shifted by
    '''
    ## update the lines for directionality. A negative spread means an expectation
    ## that the home team will win, thus their expected margin sign must be flipped
    ## whereas for teh away team, a spread represents their expected margin
    if is_home:
        model_line *= -1
        market_line *= -1
    ## establish errors ##
    model_error = abs(margin_measure - model_line)
    market_error = abs(margin_measure - market_line)
    ## establish an adjusted k that is more aggressive if the model was wrong ##
    if model_error < 1 or market_resist_factor == 0:
        ## if the model was right or their is no resist factor, no adjustment
        adj_k = k
    elif model_error <= market_error:
        ## if the market line was more accurate, no adj
        adj_k = k
    else:
        ## otherwise, increase k
        adj_k = k * (1 + abs(model_error - market_error) / market_resist_factor)
    ## calcualte the shift ##
    ## pd is a var in the elo model. It means the abs error
    pd_measure = model_error
    ## mult is the multiplier of k based on degree of obs vs exp
    mult_measure = math.log(max(pd_measure, 1) + 1, b)
    ## shift is mult * k
    shift = mult_measure * adj_k
    ## since mult and k are always positive, directionality must
    ## be determined. If expectation was beat, shift up and visa versa
    if model_error == 0:
        ## if expectation was exact, no shift
        shift = 0
    elif model_line > margin_measure:
        ## the expected result was greater than the actual
        ## then the shift is down for a downgrade ##
        shift *= -1
    ## return the shift ##
    return shift

def calc_weighted_avg(
    shift_array:list    
) -> float:
    '''
    Calculates a normalized weighted average for a list of shift / weight pairs
    passed

    Parameters:
    * shift_array (list): An array of shift, weight tuples

    Returns:
    * weighted_avg (float)
    '''
    ## structure ##
    product = 0
    weight = 0
    ## cycle through pairs and populate strucutre
    for pair in shift_array:
        ## check that value exists ##
        if not pd.isnull(pair[0]):
            product += pair[0] * pair[1]
            weight += pair[1]
    ## return ##
    return product / weight

def calc_weighted_shift(
    margin_array:list, model_line:float, market_line:float,
    k:(float or int), b:(float or int), market_resist_factor:(float or int),
    is_home:bool
) -> float:
    '''
    Abstraction that calculates the weighted average elo shift provided a list
    of margin/weight tuples, where margin represents the game outcome measure (ie
    MoV, pff, wepa), and the weight represents how much of the overall shift it should
    represent.

    Paramaters:
    * margin_array (list): array of margin/weight tuple pairs
    * model_line (float): the unregressed expectation of the model
    * market_line (float): the expectation of the market
    * k (float or in): model param that effects degree of shift
    * b (float or int): model param that effects certainty in outcome
    * market_resist_factor (float or int): determins how much to adjust back to
    the market when the model is wrong
    * is_home (bool): whether the home team is being adjusted (as opposed to away team)

    Returns:
    * weighted_shift (float): the weighted average shift to adjust the team by
    '''
    ## generate shifts / weight pairs ##
    shift_pairs = []
    for pair in margin_array:
        ## break out result and weight for clarity
        result = pair[0]
        weight = pair[1]
        ## calc shift
        shift = calc_shift(
            result, model_line, market_line, k,
            b,market_resist_factor, is_home
        )
        ## add with weight to shift pairs ##
        shift_pairs.append(
            (shift, weight)
        )
    ## create weighted average from shift pairs ##
    weighted_avg = calc_weighted_avg(shift_pairs)
    ## return ##
    return weighted_avg