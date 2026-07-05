import json
import pandas as pd
from .constants import CONFIG_DIR

configs = {}
for path in sorted(CONFIG_DIR.glob('*.json')):
    with open(path, 'r') as f:
        config = json.load(f)
    configs[config['prior']] = config

def prior_to_elo(
    season: int,
    value: float,
    prior_type: str,
    prior_elo_scale: float,
    mean_reverted_elo: float,
) -> float:
    '''
    Normalize one external prior value to elo.

    Parameters:
    * season (int): season for normalization lookup
    * value (float): raw prior value
    * prior_type (str): prior type for lookup
    * prior_elo_scale (float): z -> elo multiplier from nfelo config
    * mean_reverted_elo (float): the team's elo after internal mean reversion

    Returns:
    * prior_elo (float)
    '''
    ## guard against unknown prior type being passed ##
    if prior_type not in configs:
        raise KeyError('PRIOR ERROR: Unknown prior: {0}'.format(prior_type))
    config = configs[prior_type]
    ## extract the priors mean and normalization table ##
    mu = float(config['mu'])
    normalization = config['normalization']
    ## translate season to string for lookup ##
    season_str = str(season)
    ## fallback when raw value or sigma unavailable ##
    if season_str not in normalization:
        return mean_reverted_elo
    sigma = normalization.get(season_str)
    if value is None or pd.isnull(value) or sigma is None or float(sigma) <= 0:
        return mean_reverted_elo
    ## z-score then map to common elo scale ##
    z = (float(value) - mu) / float(sigma)
    scaled_elo = 1505 + z * float(prior_elo_scale)
    ## guard against invalid elo values ##
    if pd.isnull(scaled_elo) or scaled_elo <= 0:
        return mean_reverted_elo
    return scaled_elo
