## HELPERS ##
def initial_mr_factor(
    model_line:float, market_line:float, spread_delta_base:(float or int)
) -> float:
    '''
    Calculates the initial amount to regress the model line to the market by
    
    At a high level, the ethos is:
    * Regress fully if there little difference as this is viewed as rounding error
    * Decrease regression as the difference gets bigger (ie model is more opinionated)
    * Marginal decrease in regression is diminishing as very off market spreads have
    outsized potential to hurt model error on measures like mse without providing additional
    ATS performance

    Parameters:
    * model_line (float): model's expected median home MoV
    * market_line (float): market's expected median home MoV
    * spread_delta_base (float or int): config parameter determining shape of regression

    Returns:
    * mr_factor (float): the percent regress the model the market by 
    '''
    ## get the absolute spread dif ##
    spread_dif = abs(model_line - market_line)
    ## calc factor ##
    return (
        4 / (
            1 +
            (spread_delta_base * spread_dif**2)
        ) +
        spread_dif / 14
    )

def rmse_adj(
    model_line:float, market_line:float, rmse_base:(float or int),
    model_se_home:float, market_se_home:float,
    model_se_away:float, market_se_away:float,

) -> float:
    '''
    Increase regression if the model has been relatively less accurate
    at predicting either teams performance than market in the recent past

    Parameters:
    * model_line (float): model's expected median home MoV
    * market_line (float): market's expected median home MoV
    * rmse_base (float or int): config parameter determining shape of regression
    * model_se_home (float): the models rolling standard error for the home team
    * market_se_home (float): the markets rolling standard error for the home team
    * model_se_away (float): the models rolling standard error for the away team
    * market_se_away (float): the markets rolling standard error for the away team

    Returns:
    * rmse_factor (float): The amount to increase regression by
    '''
    ## calculate the avg rmses
    model_rmse = (model_se_home ** (1/2) + model_se_away ** (1/2)) / 2
    market_rmse = (market_se_home ** (1/2) + market_se_away ** (1/2)) / 2
    rmse_dif = model_rmse - market_rmse
    ## return ##
    ## if the spread delta is small, we dont want to override what was good
    ## regression just because the model has been more accurate. Return 1 in this case
    if abs(model_line - market_line) > 1:
        return 1 + (rmse_dif / rmse_base)
    else:
        return 1

def long_adj(
    model_line:float, market_line:float, ll_inflator:(float or int)
) -> float:
    '''
    Inflate the regression for long lines, which the model has a harder time
    getting to

    Parameters:
    * model_line (float): model's expected median home MoV
    * market_line (float): market's expected median home MoV
    * ll_inflator (float or int): degree of deflation

    Returns:
    * ll_factor (float): the percent regress the model the market by 
    '''
    if market_line < -7.5 and model_line > market_line:
        return 1 + ll_inflator
    else:
        return 1

def hook_adj(
    model_line:float, market_line:float, hook_certainty:(float or int)
) -> float:
    '''
    Analysis has shown hooks to provide value, perhaps because they signal
    uncertainty in the market's assesment of the median home MoV. As a result,
    these are regressed less

    Parameters:
    * model_line (float): model's expected median home MoV
    * market_line (float): market's expected median home MoV
    * hook_certainty (float or int): degree of deflation

    Returns:
    * hook_factor (float): the percent regress the model the market by 
    '''
    if market_line == round(market_line):
        return 1
    else:
        return 1 + hook_certainty

def regress_to_market(
    ## elo difs ##
    model_dif:float, market_dif:float,
    ## spreads ##
    model_line:float, market_line:float,
    ## config params ##
    market_regression:(float),
    min_regression:(float),
    spread_delta_base:(float or int),
    rmse_base:(float or int),
    ll_inflator:(float or int),
    hook_certainty:(float or int),
    ## error context ##
    model_se_home:float, market_se_home:float,
    model_se_away:float, market_se_away:float
):
    '''
    Wrapper for the market regression model. See underlying utilities (
    initial_mr_factor(), rmse_adj(), long_adj(), and hook_adj() for details)

    Parameters:
    * model_dif (float): model's elo difference
    * market_dif (float): market's elo difference
    * model_line (float): model's expected median home MoV
    * market_line (float): market's expected median home MoV
    * market_regression (float): baseline market regression
    * min_regression (float): minimum amount of regression 
    * spread_delta_base (float or int): config parameter determining shape of regression
    * rmse_base (float or int): config parameter determining shape of regression
    * ll_inflator (float or int): degree of deflation
    * hook_certainty (float or int): degree of deflation

    Returns:
    * regressed_dif (float): An elo dif regressed to the market
    * mr_factor_used (float): The amount of regression used
    '''
    ## set regression factors ##
    mr_factor = initial_mr_factor(model_line, market_line, spread_delta_base)
    rmse_mod = rmse_adj(
        market_line,market_line,rmse_base,
        model_se_home, market_se_home,
        model_se_away, market_se_away    
    )
    ll_mod = long_adj(model_line, market_line, ll_inflator)
    hook_mod = hook_adj(model_line, market_line, hook_certainty)
    ## combine ##
    mr_factor = mr_factor * rmse_mod * ll_mod * hook_mod
    ## translate into market regression ##
    regression_factor_used = max(
        min_regression,
        min(1, market_regression * mr_factor)
    )
    ## regress ##
    regressed_dif = model_dif + regression_factor_used * (market_dif - model_dif)
    ## return ##
    return regressed_dif, regression_factor_used
