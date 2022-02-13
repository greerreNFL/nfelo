## packages ##
import pandas as pd
import numpy
import math
import pathlib
import json
import statsmodels.api as sm

## get package directory ##
package_dir = pathlib.Path(__file__).parent.parent.resolve()

## load config file ##
config = None
with open('{0}/config.json'.format(package_dir), 'r') as fp:
    config = json.load(fp)


elo_loc = config['models']['spreads']['elo_loc']
spread_loc = config['models']['spreads']['spread_loc']
game_loc = config['models']['spreads']['game_loc']
qb_loc = config['models']['nfelo']['qb_loc']
dvoa_loc = config['models']['nfelo']['dvoa_loc']
margin_loc = config['models']['nfelo']['margin_loc']
multiples_loc = config['models']['nfelo']['multiples_loc']
spread_translation_loc = config['models']['nfelo']['spread_translation_loc']
key_nums = config['models']['nfelo']['key_nums']
nfelo_config = config['models']['nfelo']['nfelo_config']
elo_config = config['models']['spreads']['elo_config']
hfa_loc = config['models']['spreads']['hfa_loc']
nfelo_version = config['models']['nfelo']['version']



output_folder = '/output_data'


## helper funcs ##
## save last picks ##
def save_previous_picks(package_dir, output_folder, nfelo_version):
    print('     Saving last weeks picks...')
    historic_picks = None
    last_picks = None
    try:
        historic_picks = pd.read_csv(
            '{0}{1}/historic_projected_spreads.csv'.format(
                package_dir,
                output_folder
            ),
            index_col=0
        )
    except:
        print('          Couldnt find historic picks...')
    try:
        last_picks = pd.read_csv(
            '{0}{1}/projected_spreads.csv'.format(
                package_dir,
                output_folder
            ),
            index_col=0
        )
        last_picks['nfelo_version'] = nfelo_version
    except:
        print('          Couldnt find last weeks picks...')
    if historic_picks is None:
        if last_picks is None:
            pass
        else:
            last_picks.to_csv(
                '{0}{1}/historic_projected_spreads.csv'.format(
                    package_dir,
                    output_folder
                )
            )
    else:
        if last_picks is None:
            pass
        else:
            new_ids = last_picks['game_id'].to_list()
            historic_picks = historic_picks[
                ~numpy.isin(
                    historic_picks['game_id'],
                    new_ids
                )
            ]
            output = pd.concat([
                historic_picks,
                last_picks
            ])
            output.to_csv(
                '{0}{1}/historic_projected_spreads.csv'.format(
                    package_dir,
                    output_folder
                )
            )


## helper function that returns loss, push, cover probs ##
def calc_probs_favorite(proj_spread, market_spread, dist_df):
    ## flips signs of spreads ##
    proj_spread = -1 * round(proj_spread, 1)
    market_spread = -1 * round(market_spread, 1)
    ## get the probs for the nfelo projected spread ##
    temp_df = dist_df.copy()
    temp_df = temp_df[temp_df['spread_line'] == proj_spread]
    temp_df['loss_prob'] = numpy.where(
        temp_df['result'] < market_spread,
        temp_df['normalized_modeled_prob'],
        0
    )
    temp_df['push_prob'] = numpy.where(
        temp_df['result'] == market_spread,
        temp_df['normalized_modeled_prob'],
        0
    )
    temp_df['cover_prob'] = numpy.where(
        temp_df['result'] > market_spread,
        temp_df['normalized_modeled_prob'],
        0
    )
    return [
        temp_df['loss_prob'].sum(),
        temp_df['push_prob'].sum(),
        temp_df['cover_prob'].sum()
    ]


## helper function that returns loss, push, cover probs ##
def calc_probs_dog(proj_spread, market_spread, dist_df):
    ## flips signs of spreads ##
    proj_spread = -1 * round(proj_spread, 1)
    market_spread = -1 * round(market_spread, 1)
    ## get the probs for the nfelo projected spread ##
    temp_df = dist_df.copy()
    temp_df = temp_df[temp_df['spread_line'] == proj_spread]
    temp_df['loss_prob'] = numpy.where(
        temp_df['result'] < market_spread,
        temp_df['normalized_modeled_prob'],
        0
    )
    temp_df['push_prob'] = numpy.where(
        temp_df['result'] == market_spread,
        temp_df['normalized_modeled_prob'],
        0
    )
    temp_df['cover_prob'] = numpy.where(
        temp_df['result'] > market_spread,
        temp_df['normalized_modeled_prob'],
        0
    )
    return [
        temp_df['loss_prob'].sum(),
        temp_df['push_prob'].sum(),
        temp_df['cover_prob'].sum()
    ]


def get_current_week(spread_df, qb_df, game_df):
    ## minor formatting and merging ##
    spread_df = spread_df[[
        'game_id',
        'season',
        'week',
        'home_team',
        'away_team',
        'home_line_open',
        'home_line_close',
        'home_ats_pct',
        'neutral_field',
        'divisional_game',
        'market_implied_elo_dif'
    ]].copy()
    ## add margin to be able to id the most recent game ##
    spread_df = pd.merge(
        spread_df,
        game_df[[
            'game_id','result', 'old_game_id', 'gameday', 'weekday', 'gametime',
            'home_surface_advantage', 'home_time_advantage', 'home_temp_advantage'
        ]].copy(),
        on=['game_id'],
        how='left'
    )
    ## add qb_adj ##
    spread_df = pd.merge(
        spread_df,
        qb_df.drop(columns=['season', 'home_team', 'away_team']),
        on=['game_id'],
        how='left'
    )
    ## get next unplayed week ##
    unplayed_season = spread_df[numpy.isnan(spread_df['result'])]['season'].min()
    unplayed_week = spread_df[numpy.isnan(spread_df['result'])]['week'].min()
    unplayed_df = spread_df[
        (spread_df['season'] == unplayed_season) &
        (spread_df['week'] == unplayed_week)
    ].copy()
    unplayed_df['type'] = numpy.where(
        unplayed_df['season'] > 2020,
        numpy.where(
            unplayed_df['week']>18,
            'post',
            'reg'
        ),
        numpy.where(
            unplayed_df['week']>17,
            'post',
            'reg'
        )
    )
    return unplayed_df


## pulls most recent week of nfelo values and model accuracy ##
def get_most_recent_elo_values(elo_df):
    most_recent_elo_df = None
    elo_df = elo_df[[
        'game_id', 'season', 'week', 'home_team', 'away_team',
        'ending_nfelo_home', 'ending_nfelo_away',
        ## data needed for market regression ##
        'ending_model_se_away', 'ending_model_se_home',
        'ending_market_se_away', 'ending_market_se_home'
    ]].rename(columns={
        'ending_nfelo_home' : 'nfelo_home',
        'ending_nfelo_away' : 'nfelo_away',
    })
    ## gen flat file ##
    flat_elo_df = pd.concat([
        elo_df.copy()[[
            'game_id', 'week', 'home_team',
            'nfelo_home', 'ending_model_se_home', 'ending_market_se_home'
        ]].rename(columns={
            'home_team' : 'team',
            'nfelo_home' :'nfelo',
            'ending_model_se_home' : 'model_se',
            'ending_market_se_home' : 'market_se',
        }),
        elo_df.copy()[[
            'game_id', 'week', 'away_team',
            'nfelo_away', 'ending_model_se_away', 'ending_market_se_away'
        ]].rename(columns={
            'away_team' : 'team',
            'nfelo_away' :'nfelo',
            'ending_model_se_away' : 'model_se',
            'ending_market_se_away' : 'market_se',
        }),
    ])
    for team in flat_elo_df['team'].unique():
        last_game = flat_elo_df[flat_elo_df['team'] == team]['game_id'].max()
        temp_elo_df = flat_elo_df[(flat_elo_df['team'] == team) & (flat_elo_df['game_id'] == last_game)]
        if most_recent_elo_df is None:
            most_recent_elo_df = temp_elo_df.copy()
        else:
            most_recent_elo_df = pd.concat([most_recent_elo_df,temp_elo_df])
    return most_recent_elo_df

## pulls flat file of final season elos ##
def get_last_recent_elo_values(elo_df):
    last_elo_df = None
    elo_df = elo_df[[
        'game_id', 'season', 'week', 'home_team', 'away_team',
        'ending_nfelo_home', 'ending_nfelo_away',
        ## data needed for market regression ##
        'ending_model_se_away', 'ending_model_se_home',
        'ending_market_se_away', 'ending_market_se_home'
    ]].rename(columns={
        'ending_nfelo_home' : 'nfelo_home',
        'ending_nfelo_away' : 'nfelo_away',
    })
    ## gen flat file ##
    flat_elo_df = pd.concat([
        elo_df.copy()[[
            'game_id', 'week', 'home_team', 'season',
            'nfelo_home', 'ending_model_se_home', 'ending_market_se_home'
        ]].rename(columns={
            'home_team' : 'team',
            'nfelo_home' :'nfelo',
            'ending_model_se_home' : 'model_se',
            'ending_market_se_home' : 'market_se',
        }),
        elo_df.copy()[[
            'game_id', 'week', 'away_team', 'season',
            'nfelo_away', 'ending_model_se_away', 'ending_market_se_away'
        ]].rename(columns={
            'away_team' : 'team',
            'nfelo_away' :'nfelo',
            'ending_model_se_away' : 'model_se',
            'ending_market_se_away' : 'market_se',
        }),
    ])
    last_elo_df = flat_elo_df.sort_values(
        by=['team', 'season', 'week'],
        ascending=[True,True,True]
    ).reset_index(drop=True)
    last_elo_df = last_elo_df.groupby(['team', 'season']).tail(1)
    return last_elo_df





def add_lines(unplayed_df, most_recent_elo_df, hfa_df, dvoa_projections, nfelo_config, spread_mult_dict, spread_translation_dict, last_elo_df):
    ## sub function to apply info ##
    def add_info(row, config_nfelo):
        ## pull in most recent nfelo values ##
        row['home_nfelo_elo'] = most_recent_elo_df.loc[most_recent_elo_df['team'] == row['home_team']]['nfelo'].values[0]
        row['away_nfelo_elo'] = most_recent_elo_df.loc[most_recent_elo_df['team'] == row['away_team']]['nfelo'].values[0]
        ## get most recent hfa vallue ##
        current_hfa = hfa_df[
            ~pd.isnull(hfa_df['rolling_hfa'])
        ].tail(1).iloc[0]['rolling_hfa']
        ## determine base HFA ##
        if row['neutral_field'] == 1:
            row['hfa_mod'] = 1
        else:
            row['hfa_mod'] = current_hfa * 25
        ## determine divisional mod ##
        if row['divisional_game'] == 1:
            row['div_mod'] = config_nfelo['hfa_div'] * row['hfa_mod']
        else:
            row['div_mod'] = config_nfelo['hfa_non_div'] * row['hfa_mod']
        ## determine bye advantages ##
        row['home_bye_mod'] = 0
        row['away_bye_mod'] = 0
        home_last_week = most_recent_elo_df.loc[most_recent_elo_df['team'] == row['home_team']]['week'].values[0]
        away_last_week = most_recent_elo_df.loc[most_recent_elo_df['team'] == row['away_team']]['week'].values[0]
        if row['week'] == 1:
            pass
        else:
            if row['week'] > home_last_week + 1:
                row['home_bye_mod'] = row['hfa_mod'] * config_nfelo['home_bye_week']
            else:
                pass
            if row['week'] > away_last_week + 1:
                row['away_bye_mod'] = row['hfa_mod'] * config_nfelo['away_bye_week']
            else:
                pass
        ## determine field and time mods ##
        row['surface_mod'] = row['home_surface_advantage'] * nfelo_config['dif_surface'] * row['hfa_mod']
        row['time_mod'] = row['home_time_advantage'] * nfelo_config['time_advantage'] * row['hfa_mod']
        ## determine SoS rergession is (if necessary) ##
        home_fbo_proj = dvoa_projections.loc[(dvoa_projections['team'] == row['home_team']) & (dvoa_projections['season'] == row['season'])]['projected_total_dvoa'].values[0]
        away_fbo_proj = dvoa_projections.loc[(dvoa_projections['team'] == row['away_team']) & (dvoa_projections['season'] == row['season'])]['projected_total_dvoa'].values[0]
        if row['week'] == 1:
            ## get last years median elo for regression around propper mid point ##
            last_year_df = last_elo_df[
                last_elo_df['season'] == row['season'] -1
            ].copy()
            last_years_median_elo = 1505
            if len(last_year_df) == 0:
                pass
            else:
                print('          Adjusting median nfelo for YoY regression...')
                last_years_median_elo = last_year_df['nfelo'].median()
                print('               Last years median nfelo was {0}...'.format(
                    last_years_median_elo
                ))
            print('          {0}:'.format(row['home_team']))
            print('               nfelo: {0}'.format(row['home_nfelo_elo']))
            print('               adj nfelo: {0}'.format(1505 + (row['home_nfelo_elo']-last_years_median_elo)))
            print('          {0}:'.format(row['away_team']))
            print('               nfelo: {0}'.format(row['away_nfelo_elo']))
            print('               adj nfelo: {0}'.format(1505 + (row['away_nfelo_elo']-last_years_median_elo)))
            row['home_nfelo_elo'] = (
                (
                    (
                        (1505 * config_nfelo['reversion']) +
                        (
                            (
                                1505 + (row['home_nfelo_elo'] - last_years_median_elo)
                            ) * (1-config_nfelo['reversion'])
                        )
                    ) *
                    (1-config_nfelo['dvoa_weight'])
                ) +
                (
                    config_nfelo['dvoa_weight'] *
                    (1505 + 484 * home_fbo_proj)
                )
            )
            row['away_nfelo_elo'] = (
                (
                    (
                        (1505 * config_nfelo['reversion']) +
                        (
                            (
                                1505 + (row['away_nfelo_elo'] - last_years_median_elo)
                            ) * (1-config_nfelo['reversion'])
                        )
                    ) *
                    (1-config_nfelo['dvoa_weight'])
                ) +
                (
                    config_nfelo['dvoa_weight'] *
                    (1505 + 484 * away_fbo_proj)
                )
            )
        else:
            pass
        ## qb adj ##
        ## check for missing values ##
        if pd.isnull(row['home_538_qb_adj']):
            print('          WARNING: {0} was missing a QB adj. Repull from 538 or fill manually...'.format(
                row['home_team']
            ))
            row['home_538_qb_adj'] = 0
        else:
            pass
        if pd.isnull(row['away_538_qb_adj']):
            print('          WARNING: {0} was missing a QB adj. Repull from 538 or fill manually...'.format(
                row['away_team']
            ))
            row['away_538_qb_adj'] = 0
        else:
            pass
        home_net_qb_adj = config_nfelo['qb_weight'] * (row['home_538_qb_adj'] - row['away_538_qb_adj'])
        elo_dif_nfelo = (
            ## base elo difference ##
            row['home_nfelo_elo'] - row['away_nfelo_elo'] +
            ## impirical mods ##
            row['hfa_mod'] + row['home_bye_mod'] + row['away_bye_mod'] +
            row['surface_mod'] + row['time_mod'] + row['div_mod'] +
            ## QB adjustment ##
            nfelo_config['qb_weight'] * (row['home_538_qb_adj']-row['away_538_qb_adj'])
        )
        ## add playoff premium (if neccesary)
        if row['type'] == 'post':
            elo_dif_nfelo = elo_dif_nfelo * (1 + config_nfelo['playoff_boost'])
        else:
            pass
        row['home_dif_pre_reg'] = elo_dif_nfelo
        ## line pre regression ##
        home_probability_pre_regression = 1.0 / (math.pow(10.0, (-elo_dif_nfelo/config_nfelo['z'])) + 1.0)
        row['home_line_pre_regression'] = (
            ## look up spread multiplier ##
            ## spread_mult_dict[round(row['nfelo_home_probability_pre_regression'],3)] *
            -16 *
            ## multiply by a s
            math.log10(home_probability_pre_regression / max(1-home_probability_pre_regression,.001))
        )
        ## calculate market regression factor using relative errors ##
        home_se_market = most_recent_elo_df.loc[most_recent_elo_df['team'] == row['home_team']]['market_se'].values[0]
        away_se_market = most_recent_elo_df.loc[most_recent_elo_df['team'] == row['away_team']]['market_se'].values[0]
        home_se_model = most_recent_elo_df.loc[most_recent_elo_df['team'] == row['home_team']]['model_se'].values[0]
        away_se_model = most_recent_elo_df.loc[most_recent_elo_df['team'] == row['away_team']]['model_se'].values[0]
        rmse_dif = (
            (
                (
                    home_se_model ** (1/2) +
                    away_se_model ** (1/2)
                ) / 2
            ) - (
                (
                    home_se_market ** (1/2) +
                    away_se_market ** (1/2)
                ) / 2
            )
        )
        ## if nfelo is basically the same as the open, regress fully -- as it is further away, regress less
        ## When it is very far away, pair back the anti regression to improve brier (ie get closer to market line)
        ## while still preserving break even plays that come from being far away from the market open ##
        ## in laymans terms, being close to the open is fake CLV, while being further away is real CLV ##
        spread_delta_open = abs(row['home_line_pre_regression']-row['home_line_open'])
        mr_deflator_factor = (
            4 / (
                1 +
                (config_nfelo['spread_delta_base'] * spread_delta_open**2)
            ) +
            spread_delta_open / 14
        )
        mr_factor = mr_deflator_factor
        ## The model has a harder time getting to big spreads, so regress more on longer
        ## plays where the model favors the dog
        is_long = 0
        if row['home_line_open'] < -7.5 and row['home_line_pre_regression'] > row['home_line_open']:
            is_long = 1
        else:
            pass
        long_inflator = 1 + (is_long * config_nfelo['long_line_inflator'])
        mr_factor = mr_factor * long_inflator
        ## Hooks can present value, regress less ##
        is_hook = 1
        if row['home_line_close'] == round(row['home_line_close']):
            is_hook = 0
        hook_inflator = 1 + (is_hook * config_nfelo['hook_certainty'])
        mr_factor = mr_factor * hook_inflator
        ## if the spread delta is small, we don't want to override a good regression with rmse ##
        ## only apply rmse when the spread delta is over a certain amount ##
        if spread_delta_open > 1:
            mr_factor = mr_factor * (1 + rmse_dif / config_nfelo['rmse_base'])
        else:
            pass
        ## finally, make sure regression is above min regression amount ##
        mr_mult = max(config_nfelo['min_mr'],min(1, config_nfelo['market_regression'] * mr_factor))
        row['market_regression_factor'] = mr_mult
        ## make regression ##
        market_home_probability = spread_translation_dict[row['home_line_close']]
        market_elo_dif_close = (
            (-1 * config_nfelo['z']) *
            math.log10(
                (1/market_home_probability) -
                1
            )
        )
        elo_dif_nfelo = elo_dif_nfelo + mr_mult * (market_elo_dif_close - elo_dif_nfelo)
        ## add info to row ##
        row['regressed_dif'] = elo_dif_nfelo
        row['home_dif'] = row['home_nfelo_elo'] - row['away_nfelo_elo']
        row['home_net_qb_mod'] = home_net_qb_adj
        row['home_net_bye_mod'] = row['home_bye_mod'] + row['away_bye_mod']
        row['home_net_HFA_mod'] = (
            row['hfa_mod'] +
            row['surface_mod'] + row['time_mod'] + row['div_mod']
        )
        ## calc final probability and line ##
        row['home_probability_nfelo'] = (
            1.0 /
            (math.pow(10.0, (-elo_dif_nfelo/config_nfelo['z'])) + 1.0)
        )
        row['home_closing_line_nfelo'] = (
            ## look up spread multiplier ##
            ## spread_mult_dict[round(row['nfelo_home_probability'],3)] *
            ## the spread mult is removed bc we actually don;t want to land a whole or half number ##
            -16 *
            ## multiply by a s
            math.log10(row['home_probability_nfelo'] / max(1-row['home_probability_nfelo'],.001))
        )
        row['home_closing_line_rounded_nfelo'] = spread_mult_dict[round(row['home_probability_nfelo'],3)]
        row['nfelo_spread_delta'] = abs(row['home_closing_line_rounded_nfelo'] - row['home_line_close'])
        return row
    applied_unplayed_df = unplayed_df.apply(add_info, config_nfelo=nfelo_config, axis=1)
    return applied_unplayed_df


## calculate cover probabilities from spreads ##
def apply_probabilities(applied_unplayed_df, dist_df):
    ## sub function for applying probs ##
    def apply_probs(row):
        if row['home_closing_line_rounded_nfelo'] <= 0:
            home_probs = calc_probs_favorite(row['home_closing_line_rounded_nfelo'], row['home_line_close'], dist_df)
        else:
            home_probs = calc_probs_dog(row['home_closing_line_rounded_nfelo'], row['home_line_close'], dist_df)
        row['away_loss_prob'] = home_probs[2]
        row['away_push_prob'] = home_probs[1]
        row['away_cover_prob'] = home_probs[0]
        row['away_ev'] = (row['away_cover_prob'] - 1.1 * row['away_loss_prob']) / 1.1
        row['home_loss_prob'] = home_probs[0]
        row['home_push_prob'] = home_probs[1]
        row['home_cover_prob'] = home_probs[2]
        row['home_ev'] = (home_probs[2] - 1.1 * home_probs[0]) / 1.1
        return row
    applied_unplayed_df = applied_unplayed_df.apply(apply_probs, axis=1)
    return applied_unplayed_df


def calculate_spreads():
    print('Calculating current weeks projected spreads...')
    save_previous_picks(
        package_dir,
        output_folder,
        nfelo_version
    )
    elo_df = pd.read_csv(
        '{0}/{1}'.format(
            package_dir,
            elo_loc
        ),
        index_col=0
    )
    dvoa_projections = pd.read_csv(
        '{0}/{1}'.format(
            package_dir,
            dvoa_loc
        ),
        index_col=0
    )
    spread_df = pd.read_csv(
        '{0}/{1}'.format(
            package_dir,
            spread_loc
        ),
        index_col=0
    )
    qb_df = pd.read_csv(
        '{0}/{1}'.format(
            package_dir,
            qb_loc
        ),
        index_col=0
    )
    game_df = pd.read_csv(
        '{0}/{1}'.format(
            package_dir,
            game_loc
        ),
        index_col=0
    )
    hfa_df = pd.read_csv(
        '{0}/{1}'.format(
            package_dir,
            hfa_loc
        ),
        index_col=0
    )
    dist_df = pd.read_csv(
        '{0}/{1}'.format(
            package_dir,
            margin_loc
        ),
        index_col=0
    )
    last_elo_df = get_last_recent_elo_values(elo_df.copy())
    ## get current week ##
    ## rename columns to old names...probably want to clean up in future ##
    game_df = game_df.rename(columns={
        "game_date" : "gameday",
        "game_day" : "weekday",
    })
    game_df['result'] = game_df['home_score'] - game_df['away_score']
    print('     Getting current week games...')
    unplayed_df = get_current_week(spread_df, qb_df, game_df)
    print('     Getting most recent nfelo information...')
    most_recent_elo_df = get_most_recent_elo_values(elo_df)
    ## create multiples dict ##
    pre_dict = pd.read_csv(
        '{0}/{1}'.format(
            package_dir,
            multiples_loc
        )
    )
    pre_dict['win_prob'] = pre_dict['win_prob'].round(3)
    pre_dict['implied_spread'] = pre_dict['implied_spread'].round(3)
    spread_mult_dict = dict(zip(pre_dict['win_prob'],pre_dict['implied_spread']))
    ## pull out prob dict ##
    spread_dict = pd.read_csv(
        '{0}/{1}'.format(
            package_dir,
            spread_translation_loc
        )
    )
    spread_dict['spread'] = spread_dict['spread'].round(3)
    spread_dict['implied_win_prob'] = spread_dict['implied_win_prob'].round(3)
    spread_translation_dict = dict(zip(spread_dict['spread'],spread_dict['implied_win_prob']))
    print('     Calculating spreads...')
    applied_unplayed_df = add_lines(unplayed_df, most_recent_elo_df, hfa_df, dvoa_projections, nfelo_config, spread_mult_dict, spread_translation_dict, last_elo_df)
    print('     Calculating EVs...')
    applied_unplayed_df = apply_probabilities(applied_unplayed_df, dist_df)
    ## sort and filter for output ##
    applied_unplayed_df = applied_unplayed_df.sort_values(by=['nfelo_spread_delta'], ascending=False)
    applied_unplayed_df = applied_unplayed_df[[
        'game_id',
        'season',
        'week',
        'away_team',
        'home_team',
        'home_line_open',
        'home_line_close',
        'home_closing_line_rounded_nfelo',
        'home_line_pre_regression',
        'nfelo_spread_delta',
        'home_probability_nfelo',
        'home_nfelo_elo',
        'away_nfelo_elo',
        'home_dif',
        'home_dif_pre_reg',
        'market_regression_factor',
        'regressed_dif',
        'market_implied_elo_dif',
        'home_net_qb_mod',
        'home_net_HFA_mod',
        'home_net_bye_mod',
        'home_538_qb_adj',
        'away_538_qb_adj',
        'home_538_qb',
        'away_538_qb',
        'away_loss_prob',
        'away_push_prob',
        'away_cover_prob',
        'away_ev',
        'home_loss_prob',
        'home_push_prob',
        'home_cover_prob',
        'home_ev',
        'old_game_id',
        'gameday',
        'weekday',
        'gametime'
    ]]
    applied_unplayed_df['sort'] = applied_unplayed_df[['home_ev','away_ev']].max(axis=1)
    applied_unplayed_df = applied_unplayed_df.sort_values(by=['sort'], ascending=False)
    applied_unplayed_df.to_csv(
        '{0}{1}/projected_spreads.csv'.format(
            package_dir,
            output_folder
        )
    )
    print('     Done!')
