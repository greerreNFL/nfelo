## packages ##
import pandas as pd
import numpy
import math
import statistics
import pathlib
import json
import statsmodels.api as sm

## get package directory ##
package_dir = pathlib.Path(__file__).parent.parent.resolve()

## load config file ##
config = None
with open('{0}/config.json'.format(package_dir), 'r') as fp:
    config = json.load(fp)


current_loc = config['models']['nfelo']['current_loc']
wepa_loc = config['models']['nfelo']['wepa_loc']
qb_loc = config['models']['nfelo']['qb_loc']
dvoa_loc = config['models']['nfelo']['dvoa_loc']
margin_loc = config['models']['nfelo']['margin_loc']
multiples_loc = config['models']['nfelo']['multiples_loc']
spread_translation_loc = config['models']['nfelo']['spread_translation_loc']
teams = config['models']['nfelo']['teams']
key_nums = config['models']['nfelo']['key_nums']
rolling_window = config['models']['nfelo']['rolling_window']
nfelo_config = config['models']['nfelo']['nfelo_config']
begining_elo = config['models']['nfelo']['begining_elo']



output_folder = '/data_sources/nfelo'


## helper funcs ##
## calc rolling hfa ##
def calc_rolling_hfa(current_df):
    print('     Calculating rolling HFA...')
    hfa_df = current_df.copy()
    hfa_df['home_margin'] = hfa_df['home_score'] - hfa_df['away_score']
    hfa_df['rolling_hfa'] = hfa_df.rolling(window=nfelo_config['rolling_hfa_window'])['home_margin'].mean()
    hfa_df = hfa_df.groupby(['season', 'week']).agg(
        rolling_hfa = ('rolling_hfa', 'mean')
    ).reset_index()
    ## shift back a game to avoid adding forward data to games ##
    hfa_df['rolling_hfa'] = hfa_df['rolling_hfa'].shift(1)
    hfa_df.to_csv(
        '{0}{1}/rolling_hfa.csv'.format(
            package_dir,
            output_folder
        )
    )
    return hfa_df


## calculating f8 l8 windows ##
def add_windows(row, window_length, merged_df):
    windowed_df_window = merged_df[(merged_df['team'] == row['team']) & (merged_df['all_time_game_number'] > row['all_time_game_number'] - window_length) & (merged_df['all_time_game_number'] <= row['all_time_game_number'])]
    windowed_df_l16 = merged_df[(merged_df['team'] == row['team']) & (merged_df['all_time_game_number'] > row['all_time_game_number'] - 16) & (merged_df['all_time_game_number'] <= row['all_time_game_number'])]
    windowed_df_season = merged_df[(merged_df['team'] == row['team']) & (merged_df['season'] == row['season']) & (merged_df['all_time_game_number'] <= row['all_time_game_number'])]
    row['wepa_margin_L{0}'.format(window_length)] = windowed_df_window['wepa_margin'].sum()
    row['margin_L{0}'.format(window_length)] = windowed_df_window['margin'].sum()
    row['pff_point_margin_L{0}'.format(window_length)] = windowed_df_window['pff_point_margin'].sum()
    row['wepa_margin_L16'] = windowed_df_l16['wepa_margin'].sum()
    row['margin_L16'] = windowed_df_l16['margin'].sum()
    row['pff_point_margin_L16'] = windowed_df_l16['pff_point_margin'].sum()
    row['wepa_margin_ytd'] = windowed_df_season['wepa_margin'].sum()
    row['margin_ytd'] = windowed_df_season['margin'].sum()
    row['pff_point_margin_ytd'] = windowed_df_season['pff_point_margin'].sum()
    return row


## attached wepa info ##
def add_wepa(current_df, wepa_df):
    ## Add WEPA ##
    print('     Adding EPA & WEPA to game file...')
    ## calc regression for translating wepa to a margin ##
    ## only full seasons ##
    wepa_df_reg = wepa_df.copy()
    last_full_season = wepa_df_reg[wepa_df_reg['game_number'] >= 16]['season'].max()
    ## hard coding regression end date to maintain consistency w/ 17 game season ##
    last_full_season = 2020
    wepa_df_reg = wepa_df_reg[wepa_df_reg['season'] <= last_full_season]
    wepa_df_reg['intercept_constant'] = 1
    model = sm.OLS(wepa_df_reg['margin'], wepa_df_reg[['wepa_net', 'intercept_constant']], hasconst=True).fit()
    wepa_intercept = model.params.intercept_constant
    wepa_slope = model.params.wepa_net
    ## calc rsq to ensure its working correctly ##
    wepa_df_reg['margin_l8'] = numpy.where(
        (wepa_df_reg['game_number'] >8) &
        (wepa_df_reg['game_number'] <17),
        wepa_df_reg['margin'],
        0
    )
    wepa_df_reg['wepa_f8'] = numpy.where(
        (wepa_df_reg['game_number'] <=8),
        wepa_df_reg['wepa_net'],
        0
    )
    wepa_df_reg['margin_f8'] = numpy.where(
        (wepa_df_reg['game_number'] <=8),
        wepa_df_reg['margin'],
        0
    )
    wepa_df_reg_group = wepa_df_reg.groupby(['team','season']).agg(
        wepa_f8 = ('wepa_f8', 'sum'),
        margin_l8 = ('margin_l8', 'sum'),
        margin_f8 = ('margin_f8', 'sum'),
    ).reset_index()
    print('          WEPA RSQ through {0}: {1} (vs {2} for straight margin)'.format(
        last_full_season,
        round(sm.OLS(
            wepa_df_reg_group['margin_l8'],
            wepa_df_reg_group['wepa_f8'],
        ).fit().rsquared,5),
        round(sm.OLS(
            wepa_df_reg_group['margin_l8'],
            wepa_df_reg_group['margin_f8'],
        ).fit().rsquared,5)
    ))
    print('          WEPA RSQ since {0}: {1} (vs {2} for straight margin)'.format(
        last_full_season - 10,
        round(sm.OLS(
            wepa_df_reg_group[wepa_df_reg_group['season'] >= last_full_season - 10]['margin_l8'],
            wepa_df_reg_group[wepa_df_reg_group['season'] >= last_full_season - 10]['wepa_f8'],
        ).fit().rsquared,5),
        round(sm.OLS(
            wepa_df_reg_group[wepa_df_reg_group['season'] >= last_full_season - 10]['margin_l8'],
            wepa_df_reg_group[wepa_df_reg_group['season'] >= last_full_season - 10]['margin_f8'],
        ).fit().rsquared,5)
    ))
    ## filter wepa_df to only included needed fields ##
    wepa_df = wepa_df[[
        'game_id',
        'team',
        'opponent',
        'wepa_net',
        'wepa',
        'd_wepa_against',
        'epa',
        'epa_against',
        'epa_net'
    ]]
    ## create a flat file from the current file ##
    home_merge_df = current_df.copy()
    home_merge_df = home_merge_df[[
        'season',
        'week',
        'game_id',
        'home_team',
        'home_score',
        'away_score',
        'home_total_dvoa',
        'home_blended_dvoa',
        'home_pff_point_margin'
    ]]
    current_file_home_rename_dict = {
        'home_team' : 'team',
        'home_score' : 'points_for',
        'away_score' : 'points_against',
        'home_total_dvoa' : 'total_dvoa',
        'home_blended_dvoa' : 'blended_dvoa',
        'home_pff_point_margin' : 'pff_point_margin',
    }
    home_merge_df = home_merge_df.rename(columns=current_file_home_rename_dict)
    home_merge_df = pd.merge(home_merge_df,wepa_df,on=['game_id','team'],how='left')
    away_merge_df = current_df.copy()
    away_merge_df = away_merge_df[[
        'season',
        'week',
        'game_id',
        'away_team',
        'away_score',
        'home_score',
        'away_total_dvoa',
        'away_blended_dvoa',
        'away_pff_point_margin'
    ]]
    current_file_away_rename_dict = {
        'away_team' : 'team',
        'away_score' : 'points_for',
        'home_score' : 'points_against',
        'away_total_dvoa' : 'total_dvoa',
        'away_blended_dvoa' : 'blended_dvoa',
        'away_pff_point_margin' : 'pff_point_margin',
    }
    away_merge_df = away_merge_df.rename(columns=current_file_away_rename_dict)
    away_merge_df = pd.merge(away_merge_df,wepa_df,on=['game_id','team'],how='left')
    merged_df = pd.concat([home_merge_df,away_merge_df])
    merged_df = merged_df.sort_values(by=['season','week','game_id'])
    merged_df['margin'] = merged_df['points_for'] - merged_df['points_against']
    ## filter to only since 2009 and only played games ##
    merged_df = merged_df[
        (merged_df['season'] >= 2009) &
        (~pd.isnull(merged_df['margin']))
    ]
    return merged_df, wepa_slope, wepa_intercept


def calc_rolling_info(merged_df, current_df, wepa_slope, wepa_intercept):
    ## calculate rolling information ##
    ## L16, L8, YTD ##
    print('     Calculating rolling information...')
    ## add game numbers ##
    ## keep a version w/ post season for elo ##
    merged_df_elo = merged_df.copy()
    merged_df = merged_df.sort_values(by=['team','game_id'])
    merged_df['all_time_game_number'] = merged_df.groupby(['team']).cumcount() + 1
    merged_df['season_game_number'] = merged_df.groupby(['team','season']).cumcount() + 1
    merged_df = merged_df.rename(columns={
        'wepa_net' : 'net_wepa',
        'wepa' : 'offensive_wepa',
        'd_wepa_against' : 'defensive_wepa',
        'epa_net' : 'net_epa',
        'epa' : 'offensive_epa',
        'epa_against' : 'defensive_epa',
    })
    merged_df['wepa_margin'] = wepa_slope * merged_df['net_wepa'] + wepa_intercept
    ## add wepa to current_file for game grade ##
    merge_df_gg_home = merged_df.copy()[['game_id','team','net_wepa']].rename(columns={
        'team' : 'home_team',
        'net_wepa' : 'home_net_wepa',
    })
    merge_df_gg_home['home_net_wepa_point_margin'] = wepa_slope * merge_df_gg_home['home_net_wepa'] + wepa_intercept
    merge_df_gg_away = merged_df.copy()[['game_id','team','net_wepa']].rename(columns={
        'team' : 'away_team',
        'net_wepa' : 'away_net_wepa',
    })
    merge_df_gg_away['away_net_wepa_point_margin'] = wepa_slope * merge_df_gg_away['away_net_wepa'] + wepa_intercept
    current_df = pd.merge(current_df,merge_df_gg_home,on=['game_id','home_team'],how='left')
    current_df = pd.merge(current_df,merge_df_gg_away,on=['game_id','away_team'],how='left')
    ## filter to only since 2009 and only played games ##
    current_df = current_df[
        (current_df['season'] >= 2009) &
        (~pd.isnull(current_df['home_score']))
    ]
    ## add windows ##
    merged_df = merged_df.apply(
        add_windows,
        window_length=rolling_window,
        merged_df=merged_df,
        axis=1
    )
    return merged_df, current_df, merged_df_elo


def prep_elo_file(current_df, qb_df, hfa_df, nfelo_config):
    print('          Prepping file...')
    elo_game_df = current_df.copy()
    elo_game_df = elo_game_df[elo_game_df['season'] >= 2009]
    elo_game_df = pd.merge(elo_game_df,qb_df[[
        'game_id','home_538_qb_adj','away_538_qb_adj',
        'home_538_prob','qbelo_prob1'
    ]],on=['game_id'],how='left')
    ## if QB data is a missing, use 0 ##
    elo_game_df['home_538_qb_adj'] = elo_game_df['home_538_qb_adj'].fillna(0)
    elo_game_df['away_538_qb_adj'] = elo_game_df['away_538_qb_adj'].fillna(0)
    elo_game_df['home_margin'] = elo_game_df['home_score'] - elo_game_df['away_score']
    elo_game_df['home_projected_dvoa'] = elo_game_df['home_projected_dvoa'].fillna(value=0)
    elo_game_df['away_projected_dvoa'] = elo_game_df['away_projected_dvoa'].fillna(value=0)
    elo_game_df['home_blended_dvoa_begining'] = elo_game_df['home_blended_dvoa_begining'].fillna(value=0)
    elo_game_df['away_blended_dvoa_begining'] = elo_game_df['away_blended_dvoa_begining'].fillna(value=0)
    ## create a dummy SR margin which isnt used ##
    elo_game_df['home_net_sr'] = 0
    elo_game_df['away_net_sr'] = 0
    ## add previous game ids to use in elo lookup ##
    flat_game_file = pd.concat([elo_game_df[['game_id','home_team','season','week']],elo_game_df[['game_id','away_team','season','week']].rename(columns={'away_team' : 'home_team'})])
    flat_game_file = flat_game_file.sort_values(by=['home_team','game_id'])
    flat_game_file['game_number_home'] = flat_game_file.groupby(['home_team','season']).cumcount() + 1
    flat_game_file['all_time_game_number_home'] = flat_game_file.groupby(['home_team']).cumcount() + 1
    flat_game_file = flat_game_file.sort_values(by=['home_team','game_id'])
    flat_game_file['prev_game_id_home'] = flat_game_file.groupby(['home_team'])['game_id'].shift(1)
    flat_game_file['prev_week_home'] = flat_game_file.groupby(['home_team'])['week'].shift(1)
    flat_game_file = flat_game_file.drop(columns=['all_time_game_number_home'])
    flat_game_file_away = flat_game_file.rename(columns={
        'home_team' : 'away_team',
        'game_number_home' : 'game_number_away',
        'prev_game_id_home' : 'prev_game_id_away',
        'prev_week_home' : 'prev_week_away',
    })
    elo_game_df = pd.merge(elo_game_df,flat_game_file,on=['game_id','home_team','season', 'week'],how='left')
    elo_game_df = pd.merge(elo_game_df,flat_game_file_away,on=['game_id','away_team','season', 'week'],how='left')
    elo_game_df = elo_game_df.drop_duplicates()
    ## ADD IMPIRICAL ADJUSTMENTS UPFRONT ##
    ## this minimizes calcs and clutter in the main loop ##
    ## HFA ##
    ## add rolling hfa ##
    elo_game_df = pd.merge(
        elo_game_df,
        hfa_df,
        on=['week', 'season'],
        how='left'
    )
    ## fill blank observations with a standard 2.5 ##
    elo_game_df['rolling_hfa'] = elo_game_df['rolling_hfa'].fillna(2.5)
    elo_game_df['hfa_mod'] = numpy.where(
        elo_game_df['neutral_field'] == 1,
        0,
        numpy.where(
            elo_game_df['divisional_game'] == 1,
            nfelo_config['hfa_div'] * elo_game_df['rolling_hfa'] * 25,
            nfelo_config['hfa_non_div'] * elo_game_df['rolling_hfa'] * 25
        )
    )
    ## BYE ##
    ## home ##
    elo_game_df['home_bye_mod'] = numpy.where(
        elo_game_df['week'] == 1,
        0,
        numpy.where(
            elo_game_df['week'] > elo_game_df['prev_week_home'] + 1,
            nfelo_config['bye_week'],
            0
        )
    )
    ## away ##
    elo_game_df['away_bye_mod'] = numpy.where(
        elo_game_df['week'] == 1,
        0,
        numpy.where(
            elo_game_df['week'] > elo_game_df['prev_week_away'] + 1,
            nfelo_config['bye_week'],
            0
        )
    )
    ## playoffs ##
    elo_game_df['is_playoffs'] = numpy.where(
        elo_game_df['type'] == 'post',
        1,
        0
    )
    return elo_game_df


## create dictionary structure to hold elo info ##
def create_data_struc(elo_game_df):
    print('          Creating data structure...')
    elo_dict = {}
    for index, row in elo_game_df.iterrows():
        ## create initial keys if necessary ##
        ## home ##
        if row['home_team'] in elo_dict:
            pass
        else:
            elo_dict[row['home_team']] = {}
        if row['away_team'] in elo_dict:
            pass
        else:
            elo_dict[row['away_team']] = {}
        ## attach structure ##
        ## home ##
        elo_dict[row['home_team']][row['game_id']] = {
            'starting' : None,
            'ending' : None,
            'week' : None,
            'rolling_nfelo_adj_start' : 0,
            'rolling_nfelo_adj_end' : 0,
            'rolling_model_se_start' : 0,
            'rolling_model_se_end' : 0,
            'rolling_market_se_start' : 0,
            'rolling_market_se_end' : 0,
        }
        ## away ##
        elo_dict[row['away_team']][row['game_id']] = {
            'starting' : None,
            'ending' : None,
            'week' : None,
            'rolling_nfelo_adj_start' : 0,
            'rolling_nfelo_adj_end' : 0,
            'rolling_model_se_start' : 0,
            'rolling_model_se_end' : 0,
            'rolling_market_se_start' : 0,
            'rolling_market_se_end' : 0,
        }
    return elo_dict


## calculates the amount to shift each team in the elo model ##
def shift_calc_helper(margin_measure, line_measure, line_market, config, is_home):
    ## establish line direction ##
    if is_home:
        line = line_measure * -1
        market_line = line_market * -1
    else:
        line = line_measure
        market_line = line_market
    ## establish k ##
    ## adjust teams more if vegas line was closer than nfelo line ##
    if abs(margin_measure - line) < 1 or (config['market_resist_factor'] == 0):
        adj_k_measure = config['k']
    elif abs(line - margin_measure) <= abs(market_line - margin_measure):
        adj_k_measure = config['k']
    else:
        adj_k_measure = config['k'] * (1 + (abs(market_line - line) / config['market_resist_factor']))
    ## create shift ##
    adj_pd_measure = abs(margin_measure - line)
    adj_mult_measure = math.log(max(adj_pd_measure, 1) + 1, config['b'])
    shift_measure = adj_k_measure * adj_mult_measure
    ## establish shift direction ##
    if margin_measure - line == 0:
        shift_measure = 0
    elif margin_measure - line > 0:
        shift_measure = shift_measure
    else:
        shift_measure = -1.0 * shift_measure
    return shift_measure



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



def calc_nfelo(elo_game_df, spread_mult_dict, spread_translation_dict, elo_dict, nfelo_config, dist_df):
    print('          Generating nfelo values...')
    ## struc for final nfelos ##
    yearly_elos = {}
    ## defines func inside of func that can reference func variables ##
    def generate_nfelo(row, config):
        ## pull through elo probs and calc lines ##
        row['538_home_line_close_rounded'] = round(2 * (
            spread_mult_dict[round(row['home_538_prob'],3)]
        ), 0) / 2
        row['qbelo_home_line_close_rounded'] = round(2 * (
            spread_mult_dict[round(row['qbelo_prob1'],3)]
        ), 0) / 2
        row['538_home_line_close'] = (
            ## look up spread multiplier ##
            -16 *
            ## multiply by a s
            math.log10(row['home_538_prob'] / max(1-row['home_538_prob'],.001))
        )
        row['qbelo_home_line_close'] = (
            ## look up spread multiplier ##
            -16 *
            ## multiply by a s
            math.log10(row['qbelo_prob1'] / max(1-row['qbelo_prob1'],.001))
        )
        ## create starting and ending nfelos ##
        row['starting_nfelo_home'] = None
        row['starting_nfelo_away'] = None
        row['ending_nfelo_home'] = None
        row['ending_nfelo_away'] = None
        ## fill starting ##
        if row['game_number_home'] == 1: ## offseason regression ##
            if row['season'] == 2009: ## if first season, use historic starting elo ##
                row['starting_nfelo_home'] = begining_elo[row['home_team']]
            else:
                ## get last years median elo ##
                last_years_median_elo = statistics.median(yearly_elos[row['season']-1])
                row['starting_nfelo_home'] = (
                    ## Normal Elo reversion
                    (1 - config['dvoa_weight']) * (config['reversion'] * 1505 + (1 - config['reversion']) * (
                        1505 + (
                            elo_dict[row['home_team']][row['prev_game_id_home']]['ending'] -
                            last_years_median_elo
                        )
                    )) +
                    ## Blend in FBO Prior ##
                    (config['dvoa_weight'])     * (1505 + 484 * row['home_projected_dvoa'])
                )
        else:
            row['starting_nfelo_home'] = elo_dict[row['home_team']][row['prev_game_id_home']]['ending']
        if row['game_number_away'] == 1:
            if row['season'] == 2009:
                row['starting_nfelo_away'] = begining_elo[row['away_team']]
            else:
                last_years_median_elo = statistics.median(yearly_elos[row['season']-1])
                row['starting_nfelo_away'] = (
                    ## Normal Elo reversion ##
                    (1 - config['dvoa_weight']) * (config['reversion'] * 1505 + (1 - config['reversion']) * (
                        1505 + (
                            elo_dict[row['away_team']][row['prev_game_id_away']]['ending'] -
                            last_years_median_elo
                        )
                    )) +
                    ## Blend in FBO Prior ##
                    (config['dvoa_weight'])     * (1505 + 484 * row['away_projected_dvoa'])
                )
        else:
            row['starting_nfelo_away'] = elo_dict[row['away_team']][row['prev_game_id_away']]['ending']
        ## with begining Elos, calculate difference before market adj ##
        elo_diff_pre_market = (
            ## base elo difference ##
            row['starting_nfelo_home'] - row['starting_nfelo_away'] +
            ## impirical mods ##
            row['hfa_mod'] + row['home_bye_mod'] - row['away_bye_mod'] +
            ## QB adjustment ##
            config['qb_weight'] * (row['home_538_qb_adj']-row['away_538_qb_adj'])
        )
        ## add boost for playoffs where favorites play stronger ##
        ## https://fivethirtyeight.com/methodology/how-our-nfl-predictions-work/ ##
        if row['is_playoffs'] == 1:
            elo_diff_pre_market = elo_diff_pre_market * (1 + config['playoff_boost'])
        else:
            pass
        ## calculate win probability and spreads before market adj ##
        row['nfelo_home_probability_pre_market'] = 1.0 / (math.pow(10.0, (-elo_diff_pre_market/config['z'])) + 1.0)
        row['nfelo_home_line_close_pre_market'] = spread_mult_dict[round(row['nfelo_home_probability_pre_market'],3)]
        ## Update market adjustments ##
        ## Pull starting rolling nfelo adjustments ##
        try:
            row['starting_nfelo_adj_home'] = elo_dict[row['home_team']][row['prev_game_id_home']].get('rolling_nfelo_adj_end', 0)
            row['starting_nfelo_adj_away'] = elo_dict[row['away_team']][row['prev_game_id_away']].get('rolling_nfelo_adj_end', 0)
        except:
            row['starting_nfelo_adj_home'] = 0
            row['starting_nfelo_adj_away'] = 0
        ## add in market adjs ##
        elo_dif = elo_diff_pre_market
        ## save a pre regression probability ##
        row['market_home_probability'] = spread_translation_dict[row['home_line_close']]
        row['market_home_probability_open'] = spread_translation_dict[row['home_line_open']]
        row['nfelo_home_probability_pre_regression'] = 1.0 / (math.pow(10.0, (-elo_dif/config['z'])) + 1.0)
        row['nfelo_home_line_close_pre_regression'] = row['nfelo_home_line_close_pre_market']
        ## add additional reversion based on this week's difference ##
        ## first, get rolling SE's ##
        try:
            row['starting_market_se_home'] = elo_dict[row['home_team']][row['prev_game_id_home']].get('rolling_market_se_end', 0)
            row['starting_market_se_away'] = elo_dict[row['away_team']][row['prev_game_id_away']].get('rolling_market_se_end', 0)
            row['starting_model_se_home'] = elo_dict[row['home_team']][row['prev_game_id_home']].get('rolling_model_se_end', 0)
            row['starting_model_se_away'] = elo_dict[row['away_team']][row['prev_game_id_away']].get('rolling_model_se_end', 0)
        except:
            row['starting_market_se_home'] = 0
            row['starting_market_se_away'] = 0
            row['starting_model_se_home'] = 0
            row['starting_model_se_away'] = 0
        ## then calc the scaling factor ##
        rmse_dif = (
            ((row['starting_model_se_home'] ** (1/2) + row['starting_model_se_away'] ** (1/2)) / 2) -
            ((row['starting_market_se_home'] ** (1/2) + row['starting_market_se_away'] ** (1/2)) / 2)
        )
        ## if nfelo is basically the same as the open, regress fully -- as it is further away, regress less
        ## When it is very far away, pair back the anti regression to improve brier (ie get closer to market line)
        ## while still preserving break even plays that come from being far away from the market open ##
        ## in laymans terms, being close to the open is fake CLV, while being further away is real CLV ##
        spread_delta_open = abs(row['nfelo_home_line_close_pre_regression']-row['home_line_open'])
        mr_deflator_factor = (
            4 / (
                1 +
                (config['spread_delta_base'] * spread_delta_open**2)
            ) +
            spread_delta_open / 14
        )
        mr_factor = mr_deflator_factor
        ## The model has a harder time getting to big spreads, so regress more on longer
        ## plays where the model favors the dog
        is_long = 0
        if row['home_line_open'] < -7.5 and row['nfelo_home_line_close_pre_regression'] > row['home_line_open']:
            is_long = 1
        long_inflator = 1 + (is_long * config['long_line_inflator'])
        mr_factor = mr_factor * long_inflator
        ## Hooks can present value, regress less ##
        is_hook = 1
        if row['home_line_close'] == round(row['home_line_close']):
            is_hook = 0
        hook_inflator = 1 + (is_hook * config['hook_certainty'])
        mr_factor = mr_factor * hook_inflator
        ## if the spread delta is small, we don't want to override a good regression with rmse ##
        ## only apply rmse when the spread delta is over a certain amount ##
        if spread_delta_open > 1:
            mr_factor = mr_factor * (1 + rmse_dif / config['rmse_base'])
        else:
            pass
        mr_mult = max(config['min_mr'],min(1, config['market_regression'] * mr_factor))
        row['regression_factor_used'] = mr_mult
        market_elo_dif_close = (
            (-1 * config['z']) *
            math.log10(
                (1/row['market_home_probability']) -
                1
            )
        )
        market_elo_dif_open = (
            (-1 * config['z']) *
            math.log10(
                (1/row['market_home_probability_open']) -
                1
            )
        )
        elo_dif_open = elo_dif + mr_mult * (market_elo_dif_open - elo_dif)
        elo_dif = elo_dif + mr_mult * (market_elo_dif_close - elo_dif)
        ## calculate complex line based on market adjusted elo dif ##
        row['nfelo_home_probability'] = 1.0 / (math.pow(10.0, (-elo_dif/config['z'])) + 1.0)
        row['nfelo_home_probability_open'] = 1.0 / (math.pow(10.0, (-elo_dif_open/config['z'])) + 1.0)
        row['nfelo_home_line_close'] = (
            ## the unrounded line is a simple calc with a fixed multiplier of -16 ##
            -16 *
            ## multiply by a s
            math.log10(row['nfelo_home_probability'] / max(1-row['nfelo_home_probability'],.001))
        )
        ## the rounded line uses a win prob to spread translation derived from actual moneyline probs and spreads ##
        row['nfelo_home_line_close_rounded'] = spread_mult_dict[round(row['nfelo_home_probability'],3)]
        row['nfelo_home_line_open'] = (
            -16 *
            ## multiply by a s
            math.log10(row['nfelo_home_probability_open'] / max(1-row['nfelo_home_probability_open'],.001))
        )
        row['nfelo_home_line_open_rounded'] = spread_mult_dict[round(row['nfelo_home_probability_open'],3)]
        ## calc cover probs ##
        if row['nfelo_home_line_close_rounded'] <= 0:
            home_probs = calc_probs_favorite(row['nfelo_home_line_close_rounded'], row['home_line_close'], dist_df)
        else:
            home_probs = calc_probs_dog(row['nfelo_home_line_close_rounded'], row['home_line_close'], dist_df)
        if row['nfelo_home_line_open_rounded'] <= 0:
            home_probs_open = calc_probs_favorite(row['nfelo_home_line_open_rounded'], row['home_line_open'], dist_df)
        else:
            home_probs_open = calc_probs_dog(row['nfelo_home_line_open_rounded'], row['home_line_open'], dist_df)
        if row['nfelo_home_line_close_pre_market'] <= 0:
            home_probs_unregressed = calc_probs_favorite(row['nfelo_home_line_close_pre_market'], row['home_line_close'], dist_df)
        else:
            home_probs_unregressed = calc_probs_dog(row['nfelo_home_line_close_pre_market'], row['home_line_close'], dist_df)
        row['away_loss_prob'] = home_probs[2]
        row['away_push_prob'] = home_probs[1]
        row['away_cover_prob'] = home_probs[0]
        row['away_ev'] = (row['away_cover_prob'] - 1.1 * row['away_loss_prob']) / 1.1
        row['home_loss_prob'] = home_probs[0]
        row['home_push_prob'] = home_probs[1]
        row['home_cover_prob'] = home_probs[2]
        row['home_ev'] = (home_probs[2] - 1.1 * home_probs[0]) / 1.1
        row['away_loss_prob_open'] = home_probs_open[2]
        row['away_push_prob_open'] = home_probs_open[1]
        row['away_cover_prob_open'] = home_probs_open[0]
        row['away_ev_open'] = (row['away_cover_prob_open'] - 1.1 * row['away_loss_prob_open']) / 1.1
        row['home_loss_prob_open'] = home_probs_open[0]
        row['home_push_prob_open'] = home_probs_open[1]
        row['home_cover_prob_open'] = home_probs_open[2]
        row['home_ev_open'] = (home_probs_open[2] - 1.1 * home_probs_open[0]) / 1.1
        row['away_loss_prob_unregressed'] = home_probs_unregressed[2]
        row['away_push_prob_unregressed'] = home_probs_unregressed[1]
        row['away_cover_prob_unregressed'] = home_probs_unregressed[0]
        row['away_ev_unregressed'] = (row['away_cover_prob_unregressed'] - 1.1 * row['away_loss_prob_unregressed']) / 1.1
        row['home_loss_prob_unregressed'] = home_probs_unregressed[0]
        row['home_push_prob_unregressed'] = home_probs_unregressed[1]
        row['home_cover_prob_unregressed'] = home_probs_unregressed[2]
        row['home_ev_unregressed'] = (home_probs_unregressed[2] - 1.1 * home_probs_unregressed[0]) / 1.1
        ## calculate shifts ##
        ## margin ##
        margin_shift_home = shift_calc_helper(
            row['home_margin'], row['nfelo_home_line_close_pre_regression'],
            row['home_line_close'], config, True
        )
        margin_shift_away = shift_calc_helper(
            -1 * row['home_margin'], row['nfelo_home_line_close_pre_regression'],
            row['home_line_close'], config, False
        )
        ## wepa ##
        wepa_shift_home = shift_calc_helper(
            row['home_net_wepa_point_margin'], row['nfelo_home_line_close_pre_regression'],
            row['home_line_close'], config, True
        )
        wepa_shift_away = shift_calc_helper(
            row['away_net_wepa_point_margin'], row['nfelo_home_line_close_pre_regression'],
            row['home_line_close'], config, False
        )
        ## pff ##
        pff_shift_home = shift_calc_helper(
            row['home_pff_point_margin'], row['nfelo_home_line_close_pre_regression'],
            row['home_line_close'], config, True
        )
        pff_shift_away = shift_calc_helper(
            row['away_pff_point_margin'], row['nfelo_home_line_close_pre_regression'],
            row['home_line_close'], config, False
        )
        ## apply weighted average shift ##
        weighted_shift_home = (
            ## straight margin shift ##
            (margin_shift_home * config['margin_weight']) +
            ## wepa ##
            (wepa_shift_home * config['wepa_weight']) +
            ## pff ##
            (pff_shift_home * config['pff_weight'])
        )
        weighted_shift_away = (
            ## straight margin shift ##
            (margin_shift_away * config['margin_weight']) +
            ## wepa ##
            (wepa_shift_home * config['wepa_weight']) +
            ## pff ##
            (pff_shift_away * config['pff_weight'])
        )
        ## apply weighted shift ##
        row['ending_nfelo_home'] = row['starting_nfelo_home'] + (weighted_shift_home)
        row['ending_nfelo_away'] = row['starting_nfelo_away'] + (weighted_shift_away)
        ## add new elo back to dictionary ##
        nfelo_alpha = 2 / (1 + config['nfelo_span'])
        ## away ##
        elo_dict[row['away_team']][row['game_id']]['starting'] = row['starting_nfelo_away']
        elo_dict[row['away_team']][row['game_id']]['ending'] = row['ending_nfelo_away']
        elo_dict[row['away_team']][row['game_id']]['week'] = row['week']
        elo_dict[row['away_team']][row['game_id']]['rolling_nfelo_adj_start'] = row['starting_nfelo_adj_away']
        elo_dict[row['away_team']][row['game_id']]['rolling_nfelo_adj_end'] = (
            row['starting_nfelo_adj_away'] * (1 - nfelo_alpha) +
            abs(weighted_shift_away) * (nfelo_alpha)
        )
        ## home ##
        elo_dict[row['home_team']][row['game_id']]['starting'] = row['starting_nfelo_home']
        elo_dict[row['home_team']][row['game_id']]['ending'] = row['ending_nfelo_home']
        elo_dict[row['home_team']][row['game_id']]['week'] = row['week']
        elo_dict[row['home_team']][row['game_id']]['rolling_nfelo_adj_start'] = row['starting_nfelo_adj_home']
        elo_dict[row['home_team']][row['game_id']]['rolling_nfelo_adj_end'] = (
            row['starting_nfelo_adj_home'] * (1 - nfelo_alpha) +
            abs(weighted_shift_home) * (nfelo_alpha)
        )
        ## add standard errors ##
        row['se_market'] = (row['home_margin'] + row['home_line_close']) ** 2
        row['se_model'] = (row['home_margin'] + row['nfelo_home_line_close_pre_regression']) ** 2
        se_ema_alpha = 2 / (1 + config['se_span'])
        ## away ##
        elo_dict[row['away_team']][row['game_id']]['rolling_market_se_end'] = (
            row['starting_market_se_away'] * (1 - se_ema_alpha) +
            row['se_market'] * (se_ema_alpha)
        )
        elo_dict[row['away_team']][row['game_id']]['rolling_model_se_end'] = (
            row['starting_model_se_away'] * (1 - se_ema_alpha) +
            row['se_model'] * (se_ema_alpha)
        )
        elo_dict[row['home_team']][row['game_id']]['rolling_market_se_end'] = (
            row['starting_market_se_home'] * (1 - se_ema_alpha) +
            row['se_market'] * (se_ema_alpha)
        )
        elo_dict[row['home_team']][row['game_id']]['rolling_model_se_end'] = (
            row['starting_model_se_home'] * (1 - se_ema_alpha) +
            row['se_model'] * (se_ema_alpha)
        )
        ## add accuracy ##
        row['ending_market_se_home'] = (
            row['starting_market_se_home'] * (1 - se_ema_alpha) +
            row['se_market'] * (se_ema_alpha)
        )
        row['ending_market_se_away'] = (
            row['starting_market_se_away'] * (1 - se_ema_alpha) +
            row['se_market'] * (se_ema_alpha)
        )
        row['ending_model_se_home'] = (
            row['starting_model_se_home'] * (1 - se_ema_alpha) +
            row['se_model'] * (se_ema_alpha)
        )
        row['ending_model_se_away'] = (
            row['starting_model_se_away'] * (1 - se_ema_alpha) +
            row['se_model'] * (se_ema_alpha)
        )
        ## add some code to output certain elements that we want to do further exploration on ##
        row['rmse_only_mr_factor'] = 1 * (1 + rmse_dif / config['rmse_base'])
        row['is_hook'] = is_hook
        row['is_long'] = is_long
        row['spread_delta_open'] = spread_delta_open
        row['all_in_mr_factor'] = mr_factor
        row['avg_market_se'] = (row['starting_market_se_home'] + row['starting_market_se_away'])/2
        row['avg_rolling_nfelo_adj'] = (row['starting_nfelo_adj_away'] + row['starting_nfelo_adj_home'])/2
        row['avg_qb_adj'] = (abs(row['home_538_qb_adj']) + abs(row['away_538_qb_adj'])) / 2
        row['net_qb_adj'] = row['home_538_qb_adj'] - row['away_538_qb_adj']
        ## if last week of season, attach final elo to struc ##
        if row['week'] == 17:
            if row['season'] in yearly_elos.keys():
                yearly_elos[row['season']].append(row['ending_nfelo_home'])
                yearly_elos[row['season']].append(row['ending_nfelo_away'])
            else:
                yearly_elos[row['season']] = []
                yearly_elos[row['season']].append(row['ending_nfelo_home'])
                yearly_elos[row['season']].append(row['ending_nfelo_away'])
        else:
            pass
        return row
    applied_elo_df = elo_game_df.apply(generate_nfelo, config=nfelo_config, axis=1)
    return applied_elo_df


def grade_models(applied_elo_df):
    print('          Scoring models...')
    ## defines func inside of func that can reference func variables ##
    def score_models(row):
        row['538_brier'] = None
        row['qbelo_brier'] = None
        row['nfelo_brier'] = None
        row['nfelo_unregressed_brier'] = None
        row['market_brier'] = None
        row['market_open_brier'] = None
        row['538_open_ats'] = None
        row['qbelo_open_ats'] = None
        row['nfelo_open_ats'] = None
        row['538_close_ats'] = None
        row['qbelo_close_ats'] = None
        row['nfelo_close_ats'] = None
        row['538_open_ats_break_even'] = None
        row['qbelo_open_ats_break_even'] = None
        row['538_close_ats_break_even'] = None
        row['qbelo_close_ats_break_even'] = None
        row['nfelo_open_ats_be'] = None
        row['nfelo_close_ats_be'] = None
        row['nfelo_close_ats_be_unregressed'] = None
        row['538_se'] = None
        row['qbelo_se'] = None
        row['nfelo_se'] = None
        row['nfelo_unregressed_se'] = None
        row['market_se'] = None
        row['market_open_se'] = None
        row['538_su'] = None
        row['qbelo_su'] = None
        row['nfelo_su'] = None
        row['nfelo_unregressed_su'] = None
        row['market_su'] = None
        row['market_open_su'] = None
        ## calc brier ##
        if row['home_margin'] > 0:
            row['538_brier'] = 25 - (((row['home_538_prob']*100)-100)**2)/100
            row['qbelo_brier'] = 25 - (((row['qbelo_prob1']*100)-100)**2)/100
            row['nfelo_brier'] = 25 - (((row['nfelo_home_probability']*100)-100)**2)/100
            row['nfelo_unregressed_brier'] = 25 - (((row['nfelo_home_probability_pre_market']*100)-100)**2)/100
            row['market_brier'] = 25 - (((row['market_home_probability']*100)-100)**2)/100
            row['market_open_brier'] = 25 - (((row['market_home_probability_open']*100)-100)**2)/100
        else:
            row['538_brier'] = 25 - ((((1- row['home_538_prob'])*100)-100)**2)/100
            row['qbelo_brier'] = 25 - ((((1- row['qbelo_prob1'])*100)-100)**2)/100
            row['nfelo_brier'] = 25 - ((((1- row['nfelo_home_probability'])*100)-100)**2)/100
            row['nfelo_unregressed_brier'] = 25 - ((((1- row['nfelo_home_probability_pre_market'])*100)-100)**2)/100
            row['market_brier'] = 25 - ((((1- row['market_home_probability'])*100)-100)**2)/100
            row['market_open_brier'] = 25 - ((((1- row['market_home_probability_open'])*100)-100)**2)/100
        ## calc se ##
        row['538_se'] = (row['538_home_line_close'] + row['home_margin']) ** 2
        row['qbelo_se'] = (row['qbelo_home_line_close'] + row['home_margin']) ** 2
        row['nfelo_se'] = (row['nfelo_home_line_close'] + row['home_margin']) ** 2
        row['nfelo_unregressed_se'] = (row['nfelo_home_line_close_pre_market'] + row['home_margin']) ** 2
        row['market_se'] = (row['home_line_close'] + row['home_margin']) ** 2
        row['market_open_se'] = (row['home_line_open'] + row['home_margin']) ** 2
        ## calc su ##
        if row['home_margin'] == 0:
            pass
        elif row['home_margin'] > 0:
            if row['538_home_line_close'] < 0:
                row['538_su'] = 1
            else:
                row['538_su'] = 0
            if row['qbelo_home_line_close'] < 0:
                row['qbelo_su'] = 1
            else:
                row['qbelo_su'] = 0
            if row['nfelo_home_line_close_pre_market'] < 0:
                row['nfelo_unregressed_su'] = 1
            else:
                row['nfelo_unregressed_su'] = 0
            if row['nfelo_home_line_close'] < 0:
                row['nfelo_su'] = 1
            else:
                row['nfelo_su'] = 0
            if row['home_line_close'] < 0:
                row['market_su'] = 1
            else:
                row['market_su'] = 0
            if row['home_line_open'] < 0:
                row['market_open_su'] = 1
            else:
                row['market_open_su'] = 0
        else:
            if row['538_home_line_close'] > 0:
                row['538_su'] = 1
            else:
                row['538_su'] = 0
            if row['qbelo_home_line_close'] > 0:
                row['qbelo_su'] = 1
            else:
                row['qbelo_su'] = 0
            if row['nfelo_home_line_close_pre_market'] > 0:
                row['nfelo_unregressed_su'] = 1
            else:
                row['nfelo_unregressed_su'] = 0
            if row['nfelo_home_line_close'] > 0:
                row['nfelo_su'] = 1
            else:
                row['nfelo_su'] = 0
            if row['home_line_close'] > 0:
                row['market_su'] = 1
            else:
                row['market_su'] = 0
            if row['home_line_open'] > 0:
                row['market_open_su'] = 1
            else:
                row['market_open_su'] = 0
        ## calc 538 close ats ##
        if row['538_home_line_close_rounded'] == row['home_line_close'] or row['home_margin'] == -1.0 * row['home_line_close']:
            row['538_close_ats'] = numpy.nan
        elif row['538_home_line_close_rounded'] < row['home_line_close']:
            if row['home_margin'] > -1.0 * row['home_line_close']:
                row['538_close_ats'] = 1
            else:
                row['538_close_ats'] = 0
        elif row['538_home_line_close_rounded'] > row['home_line_close']:
            if row['home_margin'] < -1.0 * row['home_line_close']:
                row['538_close_ats'] = 1
            else:
                row['538_close_ats'] = 0
        else:
            row['538_close_ats'] = numpy.nan
        ## calc 538 open ats ##
        if pd.isnull(row['home_line_open']):
            row['538_open_ats'] = numpy.nan
        elif row['538_home_line_close_rounded'] == row['home_line_open'] or row['home_margin'] == -1.0 * row['home_line_open']:
            row['538_open_ats'] = numpy.nan
        elif row['538_home_line_close_rounded'] < row['home_line_open']:
            if row['home_margin'] > -1.0 * row['home_line_open']:
                row['538_open_ats'] = 1
            else:
                row['538_open_ats'] = 0
        elif row['538_home_line_close_rounded'] > row['home_line_open']:
            if row['home_margin'] < -1.0 * row['home_line_open']:
                row['538_open_ats'] = 1
            else:
                row['538_open_ats'] = 0
        else:
            row['538_open_ats'] = numpy.nan
        ## calc 538 QB close ats ##
        if row['qbelo_home_line_close_rounded'] == row['home_line_close'] or row['home_margin'] == -1.0 * row['home_line_close']:
            row['qbelo_close_ats'] = numpy.nan
        elif row['qbelo_home_line_close_rounded'] < row['home_line_close']:
            if row['home_margin'] > -1.0 * row['home_line_close']:
                row['qbelo_close_ats'] = 1
            else:
                row['qbelo_close_ats'] = 0
        elif row['qbelo_home_line_close_rounded'] > row['home_line_close']:
            if row['home_margin'] < -1.0 * row['home_line_close']:
                row['qbelo_close_ats'] = 1
            else:
                row['qbelo_close_ats'] = 0
        else:
            row['qbelo_close_ats'] = numpy.nan
        ## calc 538 QB open ats ##
        if pd.isnull(row['home_line_open']):
            row['qbelo_open_ats'] = numpy.nan
        elif row['qbelo_home_line_close_rounded'] == row['home_line_open'] or row['home_margin'] == -1.0 * row['home_line_open']:
            row['qbelo_open_ats'] = numpy.nan
        elif row['qbelo_home_line_close_rounded'] < row['home_line_open']:
            if row['home_margin'] > -1.0 * row['home_line_open']:
                row['qbelo_open_ats'] = 1
            else:
                row['qbelo_open_ats'] = 0
        elif row['qbelo_home_line_close_rounded'] > row['home_line_open']:
            if row['home_margin'] < -1.0 * row['home_line_open']:
                row['qbelo_open_ats'] = 1
            else:
                row['qbelo_open_ats'] = 0
        else:
            row['qbelo_open_ats'] = numpy.nan
        ## calc nfelo close ats ##
        if row['home_margin'] == -1.0 * row['home_line_close']:
            row['nfelo_close_ats'] = numpy.nan
        elif row['nfelo_home_line_close_rounded'] < row['home_line_close']:
            if row['home_margin'] > -1.0 * row['home_line_close']:
                row['nfelo_close_ats'] = 1
            else:
                row['nfelo_close_ats'] = 0
        elif row['nfelo_home_line_close_rounded'] > row['home_line_close']:
            if row['home_margin'] < -1.0 * row['home_line_close']:
                row['nfelo_close_ats'] = 1
            else:
                row['nfelo_close_ats'] = 0
        else:
            row['nfelo_close_ats'] = numpy.nan
        ## calc nfelo open ats ##
        if pd.isnull(row['home_line_open']):
            row['nfelo_open_ats'] = numpy.nan
        elif row['home_margin'] == -1.0 * row['home_line_open']:
            row['nfelo_open_ats'] = numpy.nan
        elif row['nfelo_home_line_open_rounded'] < row['home_line_open']:
            if row['home_margin'] > -1.0 * row['home_line_open']:
                row['nfelo_open_ats'] = 1
            else:
                row['nfelo_open_ats'] = 0
        elif row['nfelo_home_line_open_rounded'] > row['home_line_open']:
            if row['home_margin'] < -1.0 * row['home_line_open']:
                row['nfelo_open_ats'] = 1
            else:
                row['nfelo_open_ats'] = 0
        else:
            row['nfelo_open_ats'] = numpy.nan
        ## calc nfelo close ats be ##
        if pd.isnull(row['home_line_close']):
            row['nfelo_close_ats_be'] = numpy.nan
        elif row['home_margin'] == -1.0 * row['home_line_close']:
            row['nfelo_close_ats_be'] = numpy.nan
        elif row['home_ev'] > 0:
            if row['home_margin'] > -1.0 * row['home_line_close']:
                row['nfelo_close_ats_be'] = 1
            else:
                row['nfelo_close_ats_be'] = 0
        elif row['away_ev'] > 0:
            if row['home_margin'] < -1.0 * row['home_line_close']:
                row['nfelo_close_ats_be'] = 1
            else:
                row['nfelo_close_ats_be'] = 0
        else:
            row['nfelo_close_ats_be'] = numpy.nan
        ## calc nfelo open ats be ##
        if pd.isnull(row['home_line_open']):
            row['nfelo_open_ats_be'] = numpy.nan
        elif row['home_margin'] == -1.0 * row['home_line_open']:
            row['nfelo_open_ats_be'] = numpy.nan
        elif row['home_ev_open'] > 0:
            if row['home_margin'] > -1.0 * row['home_line_open']:
                row['nfelo_open_ats_be'] = 1
            else:
                row['nfelo_open_ats_be'] = 0
        elif row['away_ev_open'] > 0:
            if row['home_margin'] < -1.0 * row['home_line_open']:
                row['nfelo_open_ats_be'] = 1
            else:
                row['nfelo_open_ats_be'] = 0
        else:
            row['nfelo_open_ats_be'] = numpy.nan
        ## calc 538 close ats be ##
        if row['538_home_line_close_rounded'] == row['home_line_close'] or row['home_margin'] == -1.0 * row['home_line_close'] or abs(row['538_home_line_close_rounded'] - row['home_line_close']) <= 1.5:
            row['538_close_ats_break_even'] = numpy.nan
        elif row['538_home_line_close_rounded'] < row['home_line_close']:
            if row['home_margin'] > -1.0 * row['home_line_close']:
                row['538_close_ats_break_even'] = 1
            else:
                row['538_close_ats_break_even'] = 0
        elif row['538_home_line_close_rounded'] > row['home_line_close']:
            if row['home_margin'] < -1.0 * row['home_line_close']:
                row['538_close_ats_break_even'] = 1
            else:
                row['538_close_ats_break_even'] = 0
        else:
            row['538_close_ats_break_even'] = numpy.nan
        ## calc 538 open ats be ##
        if pd.isnull(row['home_line_open']):
            row['538_open_ats_break_even'] = numpy.nan
        elif row['538_home_line_close_rounded'] == row['home_line_open'] or row['home_margin'] == -1.0 * row['home_line_open'] or abs(row['538_home_line_close_rounded'] - row['home_line_open']) <= 1.5:
            row['538_open_ats_break_even'] = numpy.nan
        elif row['538_home_line_close_rounded'] < row['home_line_open']:
            if row['home_margin'] > -1.0 * row['home_line_open']:
                row['538_open_ats_break_even'] = 1
            else:
                row['538_open_ats_break_even'] = 0
        elif row['538_home_line_close_rounded'] > row['home_line_open']:
            if row['home_margin'] < -1.0 * row['home_line_open']:
                row['538_open_ats_break_even'] = 1
            else:
                row['538_open_ats_break_even'] = 0
        else:
            row['538_open_ats_break_even'] = numpy.nan
        ## calc 538 QB close ats be ##
        if row['qbelo_home_line_close_rounded'] == row['home_line_close'] or row['home_margin'] == -1.0 * row['home_line_close'] or abs(row['qbelo_home_line_close_rounded'] - row['home_line_close']) <= 1.5:
            row['qbelo_close_ats_break_even'] = numpy.nan
        elif row['qbelo_home_line_close_rounded'] < row['home_line_close']:
            if row['home_margin'] > -1.0 * row['home_line_close']:
                row['qbelo_close_ats_break_even'] = 1
            else:
                row['qbelo_close_ats_break_even'] = 0
        elif row['qbelo_home_line_close_rounded'] > row['home_line_close']:
            if row['home_margin'] < -1.0 * row['home_line_close']:
                row['qbelo_close_ats_break_even'] = 1
            else:
                row['qbelo_close_ats_break_even'] = 0
        else:
            row['qbelo_close_ats_break_even'] = numpy.nan
        ## calc 538 QB open ats be ##
        if pd.isnull(row['home_line_open']):
            row['qbelo_open_ats_break_even'] = numpy.nan
        elif row['qbelo_home_line_close_rounded'] == row['home_line_open'] or row['home_margin'] == -1.0 * row['home_line_open'] or abs(row['qbelo_home_line_close_rounded'] - row['home_line_open']) <= 1.5:
            row['qbelo_open_ats_break_even'] = numpy.nan
        elif row['qbelo_home_line_close_rounded'] < row['home_line_open']:
            if row['home_margin'] > -1.0 * row['home_line_open']:
                row['qbelo_open_ats_break_even'] = 1
            else:
                row['qbelo_open_ats_break_even'] = 0
        elif row['qbelo_home_line_close_rounded'] > row['home_line_open']:
            if row['home_margin'] < -1.0 * row['home_line_open']:
                row['qbelo_open_ats_break_even'] = 1
            else:
                row['qbelo_open_ats_break_even'] = 0
        else:
            row['qbelo_open_ats_break_even'] = numpy.nan
        ## calc nfelo unregressed ats be ##
        if pd.isnull(row['home_line_close']):
            row['nfelo_close_ats_be_unregressed'] = numpy.nan
        elif row['home_margin'] == -1.0 * row['home_line_close']:
            row['nfelo_close_ats_be_unregressed'] = numpy.nan
        elif row['home_ev_unregressed'] > 0:
            if row['home_margin'] > -1.0 * row['home_line_close']:
                row['nfelo_close_ats_be_unregressed'] = 1
            else:
                row['nfelo_close_ats_be_unregressed'] = 0
        elif row['away_ev_unregressed'] > 0:
            if row['home_margin'] < -1.0 * row['home_line_close']:
                row['nfelo_close_ats_be_unregressed'] = 1
            else:
                row['nfelo_close_ats_be_unregressed'] = 0
        else:
            row['nfelo_close_ats_be_unregressed'] = numpy.nan
        return row
    applied_elo_df = applied_elo_df.apply(score_models, axis=1)
    ## export market regression information for testing/exploration ##
    ## export just the model scores ##
    applied_elo_df[[
        'season', 'week', 'home_margin', 'nfelo_home_line_close_pre_regression',
        'nfelo_home_line_close', 'nfelo_home_line_close_rounded', 'home_line_close',
        'rmse_only_mr_factor', 'is_hook', 'is_long', 'spread_delta_open', 'all_in_mr_factor',
        'nfelo_open_ats_be', 'nfelo_close_ats_be', 'nfelo_close_ats_be_unregressed'
    ]].to_csv(
        '{0}{1}/individual_market_regressions.csv'.format(
            package_dir,
            output_folder
        )
    )
    ## export just the model scores ##
    applied_elo_df[[
        'season', 'week',
        '538_brier', 'qbelo_brier', 'nfelo_brier', 'nfelo_unregressed_brier', 'market_brier', 'market_open_brier',
        '538_open_ats', 'qbelo_open_ats', 'nfelo_open_ats', '538_close_ats', 'qbelo_close_ats', 'nfelo_close_ats',
        '538_open_ats_break_even', 'qbelo_open_ats_break_even', '538_close_ats_break_even', 'qbelo_close_ats_break_even',
        'nfelo_open_ats_be', 'nfelo_close_ats_be',
        '538_se', 'qbelo_se', 'nfelo_se', 'nfelo_unregressed_se', 'market_se', 'market_open_se',
        '538_su', 'qbelo_su', 'nfelo_su', 'nfelo_unregressed_su', 'market_su', 'market_open_su',
        'home_line_close', 'qbelo_home_line_close_rounded', 'nfelo_home_line_close_rounded',
        'nfelo_close_ats_be_unregressed'
    ]].to_csv(
        '{0}{1}/scored_individual_games.csv'.format(
            package_dir,
            output_folder
        )
    )
    ## create an aggregated version ##
    scored_df = applied_elo_df.groupby([True]*len(applied_elo_df)).agg(
        f538_brier = ('538_brier', 'sum'),
        qbelo_brier =('qbelo_brier', 'sum'),
        nfelo_brier =('nfelo_brier', 'sum'),
        nfelo_unregressed_brier =('nfelo_unregressed_brier', 'sum'),
        market_brier =('market_brier', 'sum'),
        f538_open_ats =('538_open_ats', 'mean'),
        qbelo_open_ats =('qbelo_open_ats', 'mean'),
        nfelo_open_ats=('nfelo_open_ats', 'mean'),
        f538_close_ats=('538_close_ats', 'mean'),
        qbelo_close_ats=('qbelo_close_ats', 'mean'),
        nfelo_close_ats=('nfelo_close_ats', 'mean'),
        nfelo_open_ats_be=('nfelo_open_ats_be', 'mean'),
        nfelo_close_ats_be=('nfelo_close_ats_be', 'mean'),
        f538_open_ats_break_even=('538_open_ats_break_even', 'mean'),
        qbelo_open_ats_break_even=('qbelo_open_ats_break_even', 'mean'),
        f538_close_ats_break_even=('538_close_ats_break_even', 'mean'),
        qbelo_close_ats_break_even=('qbelo_close_ats_break_even', 'mean'),
        possible_plays=('home_line_close', 'count'),
        plays=('nfelo_close_ats_be', 'count'),
    ).reset_index()
    print('               538 // Brier {0} // ATS {1} // ATS Open {2} // ATS Break-Even {3} // ATS Break-Even Open {4}...'.format(
        round(scored_df['f538_brier'].max(),0),
        round(scored_df['f538_close_ats'].max(),3),
        round(scored_df['f538_open_ats'].max(),3),
        round(scored_df['f538_close_ats_break_even'].max(),3),
        round(scored_df['f538_open_ats_break_even'].max(),3)
    ))
    print('               538 qbelo // Brier {0} // ATS {1} // ATS Open {2} // ATS Break-Even {3} // ATS Break-Even Open {4}...'.format(
        round(scored_df['qbelo_brier'].max(),0),
        round(scored_df['qbelo_close_ats'].max(),3),
        round(scored_df['qbelo_open_ats'].max(),3),
        round(scored_df['qbelo_close_ats_break_even'].max(),3),
        round(scored_df['qbelo_open_ats_break_even'].max(),3)
    ))
    print('               NFElo // Brier {0} // ATS {1} // ATS Open {2} // ATS Break-Even {3} // ATS Open Break-Even {4} // Play pct {5}...'.format(
        round(scored_df['nfelo_brier'].max(),0),
        round(scored_df['nfelo_close_ats'].max(),3),
        round(scored_df['nfelo_open_ats'].max(),3),
        round(scored_df['nfelo_close_ats_be'].max(),3),
        round(scored_df['nfelo_open_ats_be'].max(),3),
        round(scored_df['plays'].max() / scored_df['possible_plays'].max(),3)
    ))
    scored_df.to_csv(
        '{0}{1}/scored_models.csv'.format(
            package_dir,
            output_folder
        )
    )
    return applied_elo_df


def compile_output(applied_elo_df, merged_df, rolling_window):
    print('     Formatting output files...')
    ## join ELOs to merged_df ##
    elo_merge_home = applied_elo_df[[
        'home_team',
        'game_id',
        ## 'ending_538_elo_home',
        'ending_nfelo_home'
    ]].rename(columns={
        'home_team' : 'team',
        ## 'ending_538_elo_home' : '538_elo',
        'ending_nfelo_home' : 'nfelo',
    })
    elo_merge_away = applied_elo_df[[
        'away_team',
        'game_id',
        ##'ending_538_elo_away',
        'ending_nfelo_away'
    ]].rename(columns={
        'away_team' : 'team',
        ##'ending_538_elo_away' : '538_elo',
        'ending_nfelo_away' : 'nfelo',
    })
    elo_merge = pd.concat([elo_merge_home,elo_merge_away])
    final_team_df = pd.merge(merged_df,elo_merge,on=['team','game_id'],how='left')
    ## temporary play filler ##
    final_team_df['offensive_plays'] = 500
    final_team_df['defensive_plays'] = 500
    final_team_df = final_team_df[[
        'team',
        'all_time_game_number',
        'season_game_number',
        'season',
        'week',
        'game_id',
        'opponent',
        'points_for',
        'points_against',
        'margin',
        'offensive_plays',
        'defensive_plays',
        'offensive_epa',
        'defensive_epa',
        'net_epa',
        'offensive_wepa',
        'defensive_wepa',
        'net_wepa',
        'wepa_margin',
        'pff_point_margin',
        'margin_ytd',
        'wepa_margin_ytd',
        'pff_point_margin_ytd',
        'total_dvoa',
        'blended_dvoa',
        'margin_L{0}'.format(rolling_window),
        'wepa_margin_L{0}'.format(rolling_window),
        'pff_point_margin_L{0}'.format(rolling_window),
        'margin_L16',
        'wepa_margin_L16',
        'pff_point_margin_L16',
        ## '538_elo',
        'nfelo'
    ]].copy()
    final_team_df.to_csv(
        '{0}{1}/team_file.csv'.format(
            package_dir,
            output_folder
        )
    )
    applied_elo_df = applied_elo_df[[
        'game_id', 'type', 'season', 'week','home_team', 'away_team',
        'home_score', 'away_score', 'home_margin', 'home_line_open', 'home_line_close', 'home_ats_pct', 'game_date',
        'game_day', 'stadium', 'stadium_id', 'roof',
        'surface', 'temperature', 'wind', 'divisional_game', 'neutral_field', 'home_total_dvoa_begining',
        'home_total_dvoa', 'home_projected_dvoa', 'home_blended_dvoa_begining', 'home_blended_dvoa',
        'away_total_dvoa_begining', 'away_total_dvoa', 'away_projected_dvoa', 'away_blended_dvoa_begining',
        'away_blended_dvoa',
        ## 'starting_538_elo_home', 'starting_538_elo_away', 'ending_538_elo_home', 'ending_538_elo_away',
        ## '538_home_probability', '538_home_line_close', '538_home_line_close_rounded',
        'starting_nfelo_home', 'starting_nfelo_away', 'ending_nfelo_home', 'ending_nfelo_away', 'nfelo_home_probability', 'nfelo_home_probability_open',
        'nfelo_home_line_close', 'nfelo_home_line_close_rounded',
        'home_net_wepa', 'away_net_wepa','home_net_wepa_point_margin','away_net_wepa_point_margin',
        'home_overall_grade', 'home_pff_point_margin','away_overall_grade', 'away_pff_point_margin',
        'home_538_qb_adj', 'away_538_qb_adj',
        '538_brier', 'qbelo_brier', 'nfelo_unregressed_brier', 'nfelo_brier', 'market_brier', '538_open_ats', 'nfelo_open_ats',
        '538_close_ats', 'nfelo_close_ats', '538_open_ats_break_even','538_close_ats_break_even',
        'home_ev', 'away_ev', 'home_ev_open', 'away_ev_open', 'hfa_mod', 'home_bye_mod', 'away_bye_mod',
        'regression_factor_used', 'away_moneyline', 'home_moneyline', 'away_spread_odds', 'home_spread_odds',
        ## additional output to delete in the final version ##
        'rmse_only_mr_factor', 'is_hook', 'nfelo_home_line_close_pre_regression',
        'avg_market_se', 'starting_market_se_home', 'starting_market_se_away', 'avg_rolling_nfelo_adj',
        'avg_qb_adj', 'net_qb_adj', 'qbelo_prob1', 'nfelo_home_probability_pre_regression', '538_home_line_close', 'qbelo_home_line_close',
        ## model accuracy datapoints needed to calcualte future spreadds ##
        'ending_model_se_home', 'ending_model_se_away',
        'ending_market_se_home', 'ending_market_se_away'
    ]].copy()
    applied_elo_df.to_csv(
        '{0}{1}/current_file_w_analytics.csv'.format(
            package_dir,
            output_folder
        )
    )
    ## most recent elo grade ##
    most_recent_team_df = None
    for team in teams:
        last_game = final_team_df[final_team_df['team'] == team]['all_time_game_number'].max()
        temp_team_df = final_team_df[(final_team_df['team'] == team) & (final_team_df['all_time_game_number'] == last_game)]
        if most_recent_team_df is None:
            most_recent_team_df = temp_team_df.copy()
        else:
            most_recent_team_df = pd.concat([most_recent_team_df,temp_team_df])
    most_recent_team_df = most_recent_team_df.sort_values(by=['team']).reset_index(drop=True)
    most_recent_team_df.to_csv(
        '{0}{1}/most_recent_team_file.csv'.format(
            package_dir,
            output_folder
        )
    )
    ## create wow and ytd data points ##
    recent_elo_home = applied_elo_df.copy()[[
        'home_team',
        'game_id',
        'season',
        'week',
        ## 'ending_538_elo_home',
        'ending_nfelo_home',
        ## 'starting_538_elo_home',
        'starting_nfelo_home',
        'home_538_qb_adj'
    ]].rename(columns={
        'home_team' : 'team',
        ## 'ending_538_elo_home' : '538_elo',
        'ending_nfelo_home' : 'nfelo',
        ## 'starting_538_elo_home' : '538_elo_starting',
        'starting_nfelo_home' : 'nfelo_starting',
        'home_538_qb_adj' : '538_qb_adj',
    })
    recent_elo_away = applied_elo_df.copy()[[
        'away_team',
        'game_id',
        'season',
        'week',
        ## 'ending_538_elo_away',
        'ending_nfelo_away',
        ## 'starting_538_elo_away',
        'starting_nfelo_away',
        'away_538_qb_adj'
    ]].rename(columns={
        'away_team' : 'team',
        ## 'ending_538_elo_away' : '538_elo',
        'ending_nfelo_away' : 'nfelo',
        ## 'starting_538_elo_away' : '538_elo_starting',
        'starting_nfelo_away' : 'nfelo_starting',
        'away_538_qb_adj' : '538_qb_adj',
    })
    recent_elo = pd.concat([recent_elo_home,recent_elo_away])
    recent_elo['nfelo'] = recent_elo['nfelo'] + recent_elo['538_qb_adj']
    recent_elo['nfelo_starting'] = recent_elo['nfelo_starting'] + recent_elo['538_qb_adj']
    most_recent_elo_df = None
    for team in teams:
        last_game = recent_elo[recent_elo['team'] == team]['game_id'].max()
        temp_elo_df = recent_elo[(recent_elo['team'] == team) & (recent_elo['game_id'] == last_game)]
        if most_recent_elo_df is None:
            most_recent_elo_df = temp_elo_df.copy()
        else:
            most_recent_elo_df = pd.concat([most_recent_elo_df,temp_elo_df])
    second_to_last_elo_df = None
    for team in teams:
        second_to_last_game = recent_elo[
            (recent_elo['team'] == team) &
            (recent_elo['game_id'] != most_recent_elo_df[most_recent_elo_df['team'] == team]['game_id'].max())
        ]['game_id'].max()
        temp_elo_df = recent_elo[(recent_elo['team'] == team) & (recent_elo['game_id'] == second_to_last_game)]
        if second_to_last_elo_df is None:
            second_to_last_elo_df = temp_elo_df.copy()
        else:
            second_to_last_elo_df = pd.concat([second_to_last_elo_df,temp_elo_df])
    beg_year_elo_df = None
    for team in teams:
        beg_game = recent_elo[(recent_elo['team'] == team) & (recent_elo['week'] == 1) ]['game_id'].max()
        temp_elo_df = recent_elo[(recent_elo['team'] == team) & (recent_elo['game_id'] == beg_game)]
        if beg_year_elo_df is None:
            beg_year_elo_df = temp_elo_df.copy()
        else:
            beg_year_elo_df = pd.concat([beg_year_elo_df,temp_elo_df])
    second_to_last_elo_df = second_to_last_elo_df.rename(columns={
        ## '538_elo' : '538_elo_t_1',
        'nfelo' : 'nfelo_t_1',
    })[[
        'team',
        ## '538_elo_t_1',
        'nfelo_t_1'
    ]]
    beg_year_elo_df = beg_year_elo_df.rename(columns={
        ## '538_elo_starting' : '538_elo_soy',
        'nfelo_starting' : 'nfelo_soy',
    })[[
        'team',
        ## '538_elo_soy',
        'nfelo_soy'
    ]]
    most_recent_elo_df = pd.merge(most_recent_elo_df,second_to_last_elo_df,on=['team'],how='left')
    most_recent_elo_df = pd.merge(most_recent_elo_df,beg_year_elo_df,on=['team'],how='left')
    ## most_recent_elo_df['538_wow_delta'] = most_recent_elo_df['538_elo'] - (most_recent_elo_df['538_elo_t_1'])
    most_recent_elo_df['nfelo_wow_delta'] = most_recent_elo_df['nfelo'] - (most_recent_elo_df['nfelo_t_1'])
    ## most_recent_elo_df['538_ytd_delta'] = most_recent_elo_df['538_elo'] - (most_recent_elo_df['538_elo_soy'])
    most_recent_elo_df['nfelo_ytd_delta'] = most_recent_elo_df['nfelo'] - (most_recent_elo_df['nfelo_soy'])
    most_recent_elo_df.to_csv(
        '{0}{1}/most_recent_elo_file.csv'.format(
            package_dir,
            output_folder
        )
    )
    print('     Done!')



def calculate_nfelo():
    print('Calculating rolling analytics and nfelo...')
    print('     Loading datasets...')
    current_df = pd.read_csv(
        '{0}/{1}'.format(
            package_dir,
            current_loc
        ),
        index_col=0
    )
    wepa_df = pd.read_csv(
        '{0}/{1}'.format(
            package_dir,
            wepa_loc
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
    dvoa_projections_df = pd.read_csv(
        '{0}/{1}'.format(
            package_dir,
            dvoa_loc
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
    pre_dict = pd.read_csv(
        '{0}/{1}'.format(
            package_dir,
            multiples_loc
        )
    )
    spread_dict = pd.read_csv(
        '{0}/{1}'.format(
            package_dir,
            spread_translation_loc
        )
    )
    ## calc rolling hfa ##
    hfa_df = calc_rolling_hfa(current_df)
    ## add wepa ##
    merged_df, wepa_slope, wepa_intercept = add_wepa(current_df, wepa_df)
    ## add windows ##
    merged_df, current_df, merged_df_elo = calc_rolling_info(merged_df, current_df, wepa_slope, wepa_intercept)
    print('     Calculating Elos...')
    ## pull out spread dict ##
    pre_dict['win_prob'] = pre_dict['win_prob'].round(3)
    pre_dict['implied_spread'] = pre_dict['implied_spread'].round(3)
    spread_mult_dict = dict(zip(pre_dict['win_prob'],pre_dict['implied_spread']))
    ## pull out prob dict ##
    spread_dict['spread'] = spread_dict['spread'].round(3)
    spread_dict['implied_win_prob'] = spread_dict['implied_win_prob'].round(3)
    spread_translation_dict = dict(zip(spread_dict['spread'],spread_dict['implied_win_prob']))
    ## prep elo file ##
    elo_game_df = prep_elo_file(current_df, qb_df, hfa_df, nfelo_config)
    ## create data struct ##
    elo_dict = create_data_struc(elo_game_df)
    ## generate nfelo ##
    applied_elo_df = calc_nfelo(elo_game_df, spread_mult_dict, spread_translation_dict, elo_dict, nfelo_config, dist_df)
    ## grade models ##
    applied_elo_df = grade_models(applied_elo_df)
    ## compile output
    compile_output(applied_elo_df, merged_df, rolling_window)
