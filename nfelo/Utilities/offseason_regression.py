import pandas as pd

def offseason_regression(
    league_elo:float,
    previous_elo:float,
    proj_dvoa:float,
    proj_wt_rating:float,
    reversion:float,
    dvoa_weight:float,
    wt_weight:float
) -> float:
    '''
    Regresses a team's elo from one season to the next by
    creating a weighted average of a mean reverted previous elo,
    projected dvoa, and wt rankings
    
    Parameters:
    * league_elo (float): the previous year's league median elo
    * previous_elo (float): the team ending elo from last year
    * proj_dvoa (float): the team's projected dvoa
    * proj_wt_rating (float): the team's projected wt rating
    * reversion (float): season over season mean reversion
    * dvoa_weight (float): the weighting of projected dvoa
    * wt_weight (float) : the weighting of wt ratings
    
    Returns:
    * new_elo (float): the team's new begining elo
    '''
    ## calculate new proj elos ##
    ## normalize the previous year elo ##
    previous_elo_norm = 1505 + (previous_elo - league_elo)
    ## calculate elos ##
    mean_reverted_elo = (
        reversion * 1505 +
        (1 - reversion) * previous_elo_norm
    )
    dvoa_elo = 1505 + 484 * proj_dvoa
    wt_elo = 1505 + 24.8 * proj_wt_rating
    ## handle missing values ##
    if pd.isnull(wt_elo):
        wt_elo = mean_reverted_elo
    if pd.isnull(dvoa_elo):
        dvoa_elo = mean_reverted_elo
    ## normalize the weights if over 1 ##
    total_config_weight = dvoa_weight + wt_weight
    if total_config_weight > 1:
        dvoa_weight = dvoa_weight / total_config_weight
        wt_weight = wt_weight / total_config_weight
    ## calc implied reversion weight ##
    reverted_weight = 1 - (dvoa_weight + wt_weight)
    ## calc the weighted avg ##
    new_elo = (
        reverted_weight * mean_reverted_elo +
        dvoa_weight * dvoa_elo +
        wt_weight * wt_elo
    )
    ## return ##
    return new_elo