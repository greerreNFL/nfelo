## packages ##
import pandas as pd
import numpy
import datetime
import pathlib
import json
import statsmodels.api as sm

## get package directory ##
package_dir = pathlib.Path(__file__).parent.parent.resolve()

## load config file ##
config = None
with open('{0}/config.json'.format(package_dir), 'r') as fp:
    config = json.load(fp)


pbp_loc = config['models']['wepa']['pbp_loc']
game_loc = config['models']['wepa']['game_loc']
weight_loc = config['models']['wepa']['weight_loc']
headers = config['models']['wepa']['headers']
pbp_team_standard_dict = config['data_pulls']['nflfastR']['team_standardization']


output_folder = '/data_sources/wepa'


## helper funcs ##
## define function for calculating wepa given a dictionary of weights ##
def wepa_grade(weight_dict, test_df):
    ## define weights ##
    ## use vectorized mapping to look up weights from a dictionary ##
    ## play style ##
    test_df['qb_rush_weight'] = numpy.where((test_df['qb_scramble'] == 1) & (test_df['fumble_lost'] != 1), 1 + test_df['season'].map(weight_dict).str[0], 1)
    test_df['neutral_second_down_rush_weight'] = numpy.where(
        (test_df['down'] == 2) &
        (test_df['play_call'] == 'Run') &
        (test_df['yardline_100'] > 20) &
        (test_df['yardline_100'] < 85) &
        ((test_df['wp'] < .90) | (test_df['wp'] > .10)) &
        (test_df['qb_scramble'] != 1) &
        (test_df['fumble_lost'] != 1) &
        (test_df['epa'] < 0),
        1 + test_df['season'].map(weight_dict).str[1],
        1
    )
    test_df['incompletion_depth_s_weight'] = 1 + numpy.where(
        (test_df['incomplete_pass'] == 1) & (test_df['interception'] != 1),
        numpy.where(numpy.isnan(test_df['season'].map(weight_dict).str[2] * (2 * (1/(1 + numpy.exp(-0.1 * test_df['air_yards'] + .75)) - 0.5))),0,(test_df['season'].map(weight_dict).str[2] * (2 * (1/(1 + numpy.exp(-0.1 * test_df['air_yards'] + .75)) - 0.5)))),
        0
    )
    ## events ##
    test_df['non_sack_fumble_weight'] = numpy.where((test_df['sack'] != 1) & (test_df['fumble_lost'] == 1), 1 + test_df['season'].map(weight_dict).str[3], 1)
    test_df['int_weight'] = numpy.where(test_df['interception'] == 1, 1 + test_df['season'].map(weight_dict).str[4], 1)
    ## contextual ##
    test_df['goalline_weight'] = numpy.where((test_df['yardline_100'] < 3) & (test_df['down'] < 4), 1 + test_df['season'].map(weight_dict).str[5], 1)
    test_df['scaled_win_prob_weight'] = 1 + (-test_df['season'].map(weight_dict).str[6] * numpy.where(test_df['wp'] <= .5, 1/(1+numpy.exp(-10*(2*test_df['wp']-0.5)))-0.5,1/(1+numpy.exp(-10*(2*(1-test_df['wp'])-0.5)))-0.5))
    ## define defensive weights ##
    ## play style ##
    test_df['d_qb_rush_weight'] = numpy.where((test_df['qb_scramble'] == 1) & (test_df['fumble_lost'] != 1), 1 + test_df['season'].map(weight_dict).str[7], 1)
    test_df['d_neutral_second_down_rush_weight'] = numpy.where(
        (test_df['down'] == 2) &
        (test_df['play_call'] == 'Run') &
        (test_df['yardline_100'] > 20) &
        (test_df['yardline_100'] < 85) &
        ((test_df['wp'] < .90) | (test_df['wp'] > .10)) &
        (test_df['qb_scramble'] != 1) &
        (test_df['fumble_lost'] != 1) &
        (test_df['epa'] < 0),
        1 + test_df['season'].map(weight_dict).str[8],
        1
    )
    test_df['d_incompletion_depth_s_weight'] = 1 + numpy.where(
        (test_df['incomplete_pass'] == 1) & (test_df['interception'] != 1),
        numpy.where(numpy.isnan(test_df['season'].map(weight_dict).str[9] * (2 * (1/(1 + numpy.exp(-0.1 * test_df['air_yards'] + .75)) - 0.5))),0,(test_df['season'].map(weight_dict).str[9] * (2 * (1/(1 + numpy.exp(-0.1 * test_df['air_yards'] + .75)) - 0.5)))),
        0
    )
    ## events ##
    test_df['d_sack_fumble_weight'] = numpy.where((test_df['sack'] == 1) & (test_df['fumble_lost'] == 1), 1 + test_df['season'].map(weight_dict).str[10], 1)
    test_df['d_int_weight'] = numpy.where(test_df['interception'] == 1, 1 + test_df['season'].map(weight_dict).str[11], 1)
    test_df['d_fg_weight'] = numpy.where(test_df['play_type'] == 'field_goal', 1 + test_df['season'].map(weight_dict).str[12], 1)
    ## contextual ##
    test_df['d_third_down_pos_weight'] = numpy.where(
        (test_df['down'] == 3) &
        (test_df['epa'] > 0),
        1 + test_df['season'].map(weight_dict).str[13],
        1
    )
    ## add weights to list to build out headers and loops ##
    weight_names = [
        'qb_rush',
        'neutral_second_down_rush',
        'incompletion_depth_s',
        'non_sack_fumble',
        'int',
        'goalline',
        'scaled_win_prob'
    ]
    d_weight_names = [
        'd_qb_rush',
        'd_neutral_second_down_rush',
        'd_incompletion_depth_s',
        'd_sack_fumble',
        'd_int',
        'd_fg',
        'd_third_down_pos'
    ]
    ## create a second list for referencing the specifc weights ##
    weight_values = []
    for weight in weight_names:
        weight_values.append('{0}_weight'.format(weight))
    ## defense ##
    d_weight_values = []
    for weight in d_weight_names:
        d_weight_values.append('{0}_weight'.format(weight))
    ## create structures for aggregation ##
    aggregation_dict = {
        'margin' : 'max', ## game level margin added to each play, so take max to get 1 ##
        'wepa' : 'sum',
        'd_wepa' : 'sum',
        'epa' : 'sum',
    }
    headers = [
        'game_id',
        'posteam',
        'defteam',
        'season',
        'game_number',
        'margin',
        'wepa',
        'd_wepa',
        'epa'
    ]
    ## dictionary to rename second half of the season metrics ##
    rename_to_last_dict = {
        'margin' : 'margin_L8',
        'wepa_net' : 'wepa_net_L8',
        'epa_net' : 'epa_net_L8',
    }
    ## disctionary to join oppoenets epa to net out ##
    rename_opponent_dict = {
        'margin' : 'margin_against',
        'wepa' : 'wepa_against',
        'd_wepa' : 'd_wepa_against',
        'epa' : 'epa_against',
    }
    ## create wepa ##
    test_df['wepa'] = test_df['epa']
    for weight in weight_values:
        test_df['wepa'] = test_df['wepa'] * test_df[weight]
    test_df['d_wepa'] = test_df['epa'] * (1 + test_df['season'].map(weight_dict).str[14])
    for weight in d_weight_values:
        test_df['d_wepa'] = test_df['d_wepa'] * test_df[weight]
    ## bound wepa to prevent extreme values from introducing volatility ##
    test_df['wepa'] = numpy.where(test_df['wepa'] > 10, 10, test_df['wepa'])
    test_df['wepa'] = numpy.where(test_df['wepa'] < -10, -10, test_df['wepa'])
    ## defense ##
    test_df['d_wepa'] = numpy.where(test_df['d_wepa'] > 10, 10, test_df['d_wepa'])
    test_df['d_wepa'] = numpy.where(test_df['d_wepa'] < -10, -10, test_df['d_wepa'])
    ## aggregate from pbp to game level ##
    game_level_df = test_df.groupby(['posteam','defteam','season','game_id','game_number']).agg(aggregation_dict).reset_index()
    game_level_df = game_level_df.sort_values(by=['posteam','game_id'])
    game_level_df = game_level_df[headers]
    ## add net epa ##
    ## create an opponent data frame ##
    game_level_opponent_df = game_level_df.copy()
    game_level_opponent_df['posteam'] = game_level_opponent_df['defteam']
    game_level_opponent_df = game_level_opponent_df.drop(columns=['defteam','season','game_number'])
    game_level_opponent_df = game_level_opponent_df.rename(columns=rename_opponent_dict)
    ## merge to main game file ##
    game_level_df = pd.merge(
        game_level_df,
        game_level_opponent_df,
        on=['posteam', 'game_id'],
        how='left'
    )
    ## calculate net wepa and apply defensive adjustment ##
    game_level_df['wepa_net'] = game_level_df['wepa'] - game_level_df['d_wepa_against']
    ## rename ##
    game_level_df = game_level_df.rename(columns={'posteam' : 'team', 'defteam' : 'opponent'})
    ## rejoin oppoenent net wepa ##
    game_level_df_opponent = game_level_df.copy()
    game_level_df_opponent = game_level_df_opponent[['opponent', 'game_id', 'wepa_net']].rename(columns={
        'opponent' : 'team',
        'wepa_net' : 'wepa_net_opponent',
    })
    game_level_df = pd.merge(
        game_level_df,
        game_level_df_opponent,
        on=['team', 'game_id'],
        how='left'
    )
    return game_level_df


def calculate_wepa():
    print('Calculating WEPA...')
    print('     Note -- if running for new season, update model first...')
    print('     Loading model weights...')
    model_weights_df = pd.read_csv(
        '{0}/{1}'.format(
            package_dir,
            weight_loc
        ),
        index_col=0
    )
    most_recent_season = int(model_weights_df['season'].max())
    print('          Found models through the {0} season...'.format(most_recent_season))
    ## struct for model weights dict ##
    weights_by_season = {}
    ## turn weight data frame into a weight dictionary ##
    for index, row in model_weights_df.iterrows():
        weights_by_season[row['season']] = [
            row['qb_rush'],    ## qb_rush ##
            row['neutral_second_down_rush'],    ## neutral_second_down_rush ##
            row['incompletion_depth_s'],    ## incompletion_depth_s ##
            row['non_sack_fumble'],    ## non_sack_fumble ##
            row['int'],    ## int ##
            row['goalline'],    ## goalline ##
            row['scaled_win_prob'],    ## scaled_win_prob ##
            row['d_qb_rush'],    ## d_qb_rush ##
            row['d_neutral_second_down_rush'],    ## d_neutral_second_down_rush ##
            row['d_incompletion_depth_s'],    ## d_incompletion_depth_s ##
            row['d_sack_fumble'],    ## d_sack_fumble ##
            row['d_int'],    ## d_int ##
            row['d_fg'],    ## d_fg ##
            row['d_third_down_pos'],    ## d_third_down_pos ##
            row['defense_adj']     ## defense_adj ##
        ]
    ## prep pbp file ##
    print('     Loading PBP data...')
    pbp_df = pd.read_csv(
        '{0}/{1}'.format(
            package_dir,
            pbp_loc
        ),
        low_memory=False,
        index_col=0
    )
    print('     Prepping PBP data...')
    ## extra poitns have an epa bug on pre 2010 data, so zero out ##
    pbp_df['epa'] = numpy.where(pbp_df['play_type'] == 'extra_point', 0, pbp_df['epa'])
    pbp_df['posteam'] = pbp_df['posteam'].replace(pbp_team_standard_dict)
    pbp_df['defteam'] = pbp_df['defteam'].replace(pbp_team_standard_dict)
    pbp_df['penalty_team'] = pbp_df['penalty_team'].replace(pbp_team_standard_dict)
    pbp_df['home_team'] = pbp_df['home_team'].replace(pbp_team_standard_dict)
    pbp_df['away_team'] = pbp_df['away_team'].replace(pbp_team_standard_dict)
    ## replace game_id using standardized franchise names ##
    pbp_df['game_id'] = (
        pbp_df['season'].astype('str') +
        '_' +
        pbp_df['week'].astype('str').str.zfill(2) +
        '_' +
        pbp_df['away_team'] +
        '_' +
        pbp_df['home_team']
    )
    ## fix some data formatting issues ##
    pbp_df['yards_after_catch'] = pd.to_numeric(pbp_df['yards_after_catch'], errors='coerce')
    ## denote pass or run ##
    ## seperate offensive and defensive penalties ##
    pbp_df['off_penalty'] = numpy.where(pbp_df['penalty_team'] == pbp_df['posteam'], 1, 0)
    pbp_df['def_penalty'] = numpy.where(pbp_df['penalty_team'] == pbp_df['defteam'], 1, 0)
    ## pandas wont group nans so must fill with a value ##
    pbp_df['penalty_type'] = pbp_df['penalty_type'].fillna('No Penalty')
    ## accepted pentalites on no plays need additional detail to determine if they were a pass or run ##
    ## infer pass plays from the play description ##
    pbp_df['desc_based_dropback'] = numpy.where(
        (
            (pbp_df['desc'].str.contains(' pass ', regex=False)) |
            (pbp_df['desc'].str.contains(' sacked', regex=False)) |
            (pbp_df['desc'].str.contains(' scramble', regex=False))
        ),
        1,
        0
    )
    ## infer run plays from the play description ##
    pbp_df['desc_based_run'] = numpy.where(
        (
            (~pbp_df['desc'].str.contains(' pass ', regex=False, na=False)) &
            (~pbp_df['desc'].str.contains(' sacked', regex=False, na=False)) &
            (~pbp_df['desc'].str.contains(' scramble', regex=False, na=False)) &
            (~pbp_df['desc'].str.contains(' kicks ', regex=False, na=False)) &
            (~pbp_df['desc'].str.contains(' punts ', regex=False, na=False)) &
            (~pbp_df['desc'].str.contains(' field goal ', regex=False, na=False)) &
            (pbp_df['desc'].str.contains(' to ', regex=False)) &
            (pbp_df['desc'].str.contains(' for ', regex=False))
        ),
        1,
        0
    )
    ## coalesce coded and infered drop backs ##
    pbp_df['qb_dropback'] = pbp_df[['qb_dropback', 'desc_based_dropback']].max(axis=1)
    ## coalesce coaded and infered rush attemps ##
    pbp_df['rush_attempt'] = pbp_df[['rush_attempt', 'desc_based_run']].max(axis=1)
    ## create a specific field for play call ##
    pbp_df['play_call'] = numpy.where(
        pbp_df['qb_dropback'] == 1,
        'Pass',
        numpy.where(
            pbp_df['rush_attempt'] == 1,
            'Run',
            numpy.nan
        )
    )
    ## Structure game file to attach to PBP data ##
    print('     Loading game file...')
    game_file_df = pd.read_csv(
        '{0}/{1}'.format(
            package_dir,
            game_loc
        ),
        index_col=0
    )
    ## calc margin ##
    game_file_df['home_margin'] = game_file_df['home_score'] - game_file_df['away_score']
    game_file_df['away_margin'] = game_file_df['away_score'] - game_file_df['home_score']
    ## flatten file to attach to single team
    game_home_df = game_file_df.copy()[['game_id', 'week', 'season', 'home_team', 'home_margin']].rename(columns={
        'home_team' : 'posteam',
        'home_margin' : 'margin',
    })
    game_away_df = game_file_df.copy()[['game_id', 'week', 'season', 'away_team', 'away_margin']].rename(columns={
        'away_team' : 'posteam',
        'away_margin' : 'margin',
    })
    flat_game_df = pd.concat([game_home_df,game_away_df], ignore_index=True).sort_values(by=['game_id'])
    ## calculate game number to split in regressions ##
    flat_game_df['game_number'] = flat_game_df.groupby(['posteam', 'season']).cumcount() + 1
    ## merge to pbp now, so you don't have to merge on every loop ##
    print('     Merging to PBP...')
    pbp_df = pd.merge(
        pbp_df,
        flat_game_df[['posteam','game_id','margin', 'game_number']],
        on=['posteam','game_id'],
        how='left'
    )
    ## calc wepa ##
    print('     Calculating WEPA...')
    wepa_df = wepa_grade(weights_by_season, pbp_df.copy())
    wepa_df = wepa_df[wepa_df['season'] > 1999]
    ## final formatting ##
    wepa_df['epa_net'] = wepa_df['epa'] - wepa_df['epa_against']
    wepa_df['epa_net_opponent'] = wepa_df['epa_against'] - wepa_df['epa']
    wepa_df = wepa_df[headers]
    print('     Exporting...')
    wepa_df.to_csv(
        '{0}{1}/wepa_flat_file.csv'.format(
            package_dir,
            output_folder
        )
    )
    print('     Done!')
