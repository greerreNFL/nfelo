from .cover_probability import calc_cover_probs

def calc_clv(original_home_spread:float, current_home_spread:float):
    '''
    Calculates the CLV for home and away teams provided a starting
    and closing home spread

    Parameters:
    * original_home_spread: the home spread at the start of the week
    * current_home_spread: the home spread at the end of the week

    Returns:
    * clv_home: the CLV for the home team
    * clv_away: the CLV for the away team
    '''
    ## get cover probabilities ##
    ## at open ##
    ## note this must be calculated to get an exact starting EV number to compare the
    ## current EV to ##
    home_loss_prob_open, home_push_prob_open, home_cover_prob_open = calc_cover_probs(
        proj_spread=original_home_spread,
        market_spread=original_home_spread
    )
    ## at close ##
    home_loss_prob, home_push_prob, home_cover_prob = calc_cover_probs(
        ## projected spread is the "true prob", while the market spread
        ## is the "priced prob". Thus, the cover prob here uses the
        ## current odds (closing line) as the true probs
        proj_spread=current_home_spread,
        market_spread=original_home_spread
    )
    ## assign to away ##
    away_loss_prob_open = home_cover_prob_open
    away_push_prob_open = home_push_prob_open
    away_cover_prob_open = home_loss_prob_open
    away_loss_prob = home_cover_prob
    away_push_prob = home_push_prob
    away_cover_prob = home_loss_prob
    ## calculate CLVs ##
    ## assume 1 for win and -1.1 for loss ##
    ## assume odds at open were a perfect 50/50 ##
    ## these need to be subtracted to reflect 0 line movement ##
    ## is 0% CLV, even if it is negative EV
    clv_home = (
        ## current nominal EV ##
        (
            home_cover_prob * 1 +
            home_push_prob * 0 +
            home_loss_prob * -1.1
        ) - 
        ## less starting nominal EV ##
        (
            home_cover_prob_open * 1 +
            home_push_prob_open * 0 +
            home_loss_prob_open * -1.1
        )
    ) / 1.1 ## div by risk to translate to a EV percentage ##
    clv_away = (
        (
            away_cover_prob * 1 +
            away_push_prob * 0 +
            away_loss_prob * -1.1
        ) - 
        (
            away_cover_prob_open * 1 +
            away_push_prob_open * 0 +
            away_loss_prob_open * -1.1
        )
    ) / 1.1 ## div by risk to translate to a EV percentage ##
    ## return clvs ##
    return clv_home, clv_away
    
    

    