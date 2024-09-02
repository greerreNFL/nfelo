import pandas as pd
import pathlib
import json

from .Data import DataLoader
from .Model import Nfelo
from .Optimizer import NfeloOptimizer

def optimize_nfelo_core():
    '''
    Optimizes the unregressed nfelo model for brier
    '''
    ## load config ##
    config_loc = '{0}/config.json'.format(
        pathlib.Path(__file__).parent.parent.resolve()
    )
    with open(config_loc, 'r') as fp:
        config = json.load(fp)
    ## load data ##
    data = DataLoader()
    nfelo = Nfelo(
        data=data,
        config=config['models']['nfelo']
    )
    ## optmize ##
    optimizer = NfeloOptimizer(
        'nfelo-core',
        nfelo,
        [
            'k', 'z', 'b', 'reversion', 'dvoa_weight',
            'wt_ratings_weight', 'margin_weight', 'pff_weight',
            'wepa_weight'
        ],
        'nfelo_brier'
    )
    optimizer.optimize()
    optimizer.save_to_logs()

def optimize_nfelo_base(return_config=False, basin_hop=False, bg_overrides={}):
    '''
    Optimizes the unregressed nfelo model for adj brier
    This starts to use market signals, so adj brier is used
    '''
    ## load config ##
    config_loc = '{0}/config.json'.format(
        pathlib.Path(__file__).parent.parent.resolve()
    )
    with open(config_loc, 'r') as fp:
        config = json.load(fp)
    ## load data ##
    data = DataLoader()
    nfelo = Nfelo(
        data=data,
        config=config['models']['nfelo']
    )
    ## optmize ##
    optimizer = NfeloOptimizer(
        'nfelo-base',
        nfelo,
        [
            'k', 'z', 'b', 'reversion', 'dvoa_weight',
            'wt_ratings_weight', 'margin_weight', 'pff_weight',
            'wepa_weight', 'market_resist_factor'
        ],
        'nfelo_brier_adj',
        basin_hop=basin_hop,
        bg_overrides=bg_overrides
    )
    optimizer.optimize()
    optimizer.save_to_logs()
    if return_config:
        return optimizer.nfelo_model.config

def optimize_nfelo_mr(pass_config=None):
    '''
    Optimize the market regression component of the nfelo model
    '''
    ## load config ##
    config_loc = '{0}/config.json'.format(
        pathlib.Path(__file__).parent.parent.resolve()
    )
    ## handle overrides from base model
    with open(config_loc, 'r') as fp:
        config = json.load(fp)
    if pass_config is None:
        ## override with fixed values #
        ## this is temp for development ##
        config['models']['nfelo']['nfelo_config']['k'] = 9.114
        config['models']['nfelo']['nfelo_config']['z'] = 401.62
        config['models']['nfelo']['nfelo_config']['b'] = 9.999
        config['models']['nfelo']['nfelo_config']['reversion'] = 0.001 
        config['models']['nfelo']['nfelo_config']['dvoa_weight'] = 0.474
        config['models']['nfelo']['nfelo_config']['wt_ratings_weight'] = 0.05
        config['models']['nfelo']['nfelo_config']['margin_weight'] = 0.6633
        config['models']['nfelo']['nfelo_config']['pff_weight'] = 0.1000
        config['models']['nfelo']['nfelo_config']['wepa_weight'] = 0.13529
        config['models']['nfelo']['nfelo_config']['market_resist_factor'] = 1.5039
    else:
        ## repack the config as it comes from an upacked version from nfelo ##
        for k,v in pass_config.items():
            config['models']['nfelo']['nfelo_config'][k] = v
    ## load data ##
    data = DataLoader()
    nfelo = Nfelo(
        data=data,
        config=config['models']['nfelo']
    )
    ## optmize ##
    optimizer = NfeloOptimizer(
        'nfelo-mr',
        nfelo,
        [
            'market_regression', 'se_span', 'rmse_base',
            'spread_delta_base', 'hook_certainty',
            'long_line_inflator', 'min_mr'
        ],
        'nfelo_brier_close',
        ## changing market params is noiser, so use
        ## slightly modified opti params ##
        basin_hop=True
    )
    optimizer.optimize()
    optimizer.save_to_logs()


def optimize_all():
    '''
    Generates an optimziation for for all 4 types and passes
    the best config from nfelo_base to the final two optimizations
    '''
    ## optimize core for context ##
    optimize_nfelo_core()
    ## optimize base ##
    new_config = optimize_nfelo_base(return_config=True)
    ## optimize ats and mr ##
    optimize_nfelo_mr(pass_config=new_config)
    return None

def optimize_base_with_var(var:str, bg_overrides:list):
    '''
    Because the objective function is not particularly smooth,
    the optimizer can get stuck on best guesses

    This func takes a variable and a set of guesses to override the
    starting best guess with
    '''
    for override in bg_overrides:
        override = {
            var : override
        }
        optimize_nfelo_base(bg_overrides=override)

def optimize_base_with_k():
    '''
    Optimize base with k bestguess overrides
    '''
    ## overrides = [8,9,15,16,17,18]
    ## optimize_base_with_var('k', overrides)
    overrides = [300,350,375,425,500]
    optimize_base_with_var('z', overrides)
    overrides = [4,5,6,7,8,9]
    optimize_base_with_var('b', overrides)
