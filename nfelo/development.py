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

def optimize_nfelo_base(return_config=False):
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
        'nfelo_brier_adj'
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
        config['models']['nfelo']['nfelo_config']['k'] = 13.5
        config['models']['nfelo']['nfelo_config']['z'] = 436.0
        config['models']['nfelo']['nfelo_config']['b'] = 7.78
        config['models']['nfelo']['nfelo_config']['reversion'] = 0.15
        config['models']['nfelo']['nfelo_config']['dvoa_weight'] = 0.433
        config['models']['nfelo']['nfelo_config']['wt_ratings_weight'] = 0.15
        config['models']['nfelo']['nfelo_config']['margin_weight'] = 0.636
        config['models']['nfelo']['nfelo_config']['pff_weight'] = 0.157
        config['models']['nfelo']['nfelo_config']['wepa_weight'] = 0.14
        config['models']['nfelo']['nfelo_config']['market_resist_factor'] = 2.24
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
        'nfelo_brier_close'
    )
    optimizer.optimize()
    optimizer.save_to_logs()


def optimize_nfelo_ats(pass_config=None):
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
        config['models']['nfelo']['nfelo_config']['k'] = 12.71
        config['models']['nfelo']['nfelo_config']['z'] = 472.95
        config['models']['nfelo']['nfelo_config']['b'] = 8.43
        config['models']['nfelo']['nfelo_config']['reversion'] = 0.188 
        config['models']['nfelo']['nfelo_config']['dvoa_weight'] = 0.39
        config['models']['nfelo']['nfelo_config']['wt_ratings_weight'] = 0.209
        config['models']['nfelo']['nfelo_config']['margin_weight'] = 0.623
        config['models']['nfelo']['nfelo_config']['pff_weight'] = 0.343
        config['models']['nfelo']['nfelo_config']['wepa_weight'] = 0.166
        config['models']['nfelo']['nfelo_config']['market_resist_factor'] = 1.4
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
        'nfelo-ats',
        nfelo,
        [
            'market_regression', 'se_span', 'rmse_base',
            'spread_delta_base', 'hook_certainty',
            'long_line_inflator', 'min_mr'
        ],
        'nfelo_brier_close_ats'
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
    optimize_nfelo_ats(pass_config=new_config)
    return None