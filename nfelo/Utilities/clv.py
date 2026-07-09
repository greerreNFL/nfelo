import math
from nfelotranslation import Translator

def calc_clv(
    original_home_spread:float,
    current_home_spread:float,
    season:int,
) -> tuple[float, float]:
    '''
    Calculates the CLV for home and away teams provided a starting
    and closing home spread. Cover/push probabilities come from the
    per-season nfelotranslation margin model — independent of nfelo's
    model projection or market regression state.

    Parameters:
    * original_home_spread: the home spread at the start of the week
      (nfelo sportsbook convention: negative = home favored)
    * current_home_spread: the home spread at the end of the week
    * season: NFL season year (selects per-season translation models)

    Returns:
    * clv_home: float: the CLV for the home team
    * clv_away: float: the CLV for the away team
    '''
    ## negate spreads at the nfelo<->nfelotranslation boundary; nfelo uses ##
    ## sportsbook (negative=home favored), nfelotranslation uses positive=home ##
    if math.isnan(original_home_spread) or math.isnan(current_home_spread):
        return float('nan'), float('nan')
    nt_open = -original_home_spread
    nt_close = -current_home_spread
    translator = Translator(nt_open, 'spread', season=int(season), side='home')
    ## at open ##
    ## note this must be calculated to get an exact starting EV number to compare the
    ## current EV to ##
    home_cover_prob_open = translator.cover_prob(nt_open)
    home_push_prob_open = translator.push_prob(nt_open)
    home_loss_prob_open = 1 - home_cover_prob_open - home_push_prob_open
    ## at close ##
    ## projected spread is the "true prob", while the market spread
    ## is the "priced prob". Thus, the cover prob here uses the
    ## current odds (closing line) as the true probs
    translator.update(nt_close, 'spread')
    home_cover_prob = translator.cover_prob(nt_open)
    home_push_prob = translator.push_prob(nt_open)
    home_loss_prob = 1 - home_cover_prob - home_push_prob
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
