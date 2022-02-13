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


game_loc = config['formatting']['game_merge']['game_loc']
spread_loc = config['formatting']['game_merge']['spread_loc']
dvoa_proj_loc = config['formatting']['game_merge']['dvoa_proj_loc']
dvoa_weekly_loc = config['formatting']['game_merge']['dvoa_weekly_loc']
pff_loc = config['formatting']['game_merge']['pff_loc']
weather_loc = config['formatting']['game_merge']['weather_loc']


pbp_team_standard_dict = config['data_pulls']['nflfastR']['team_standardization']
pbp_surface_repl = config['data_pulls']['nflfastR']['surface_repl']
pbp_timezones = config['data_pulls']['nflfastR']['timezones']
pbp_timezone_overrides = config['data_pulls']['nflfastR']['timezone_overrides']
game_headers = config['formatting']['game_merge']['game_headers']
game_repl = config['formatting']['game_merge']['game_repl']
home_dvoa_repl = config['formatting']['game_merge']['home_dvoa_repl']
away_dvoa_repl = config['formatting']['game_merge']['away_dvoa_repl']
final_headers = config['formatting']['game_merge']['final_headers']
manual_clean_dict = config['formatting']['game_merge']['manual_clean_dict']


## create output sub path ##
output_folder = '/data_sources/formatted'


## helper funcs ##
def calc_begining_dvoa(row, dvoa_new_df):
    if pd.isnull(row['week']):
        row['blended_dvoa_begining'] = row['projected_total_dvoa']
        row['total_dvoa_begining'] = numpy.nan
        row['week'] = 1
    elif row['week'] == 1:
        row['blended_dvoa_begining'] = row['projected_total_dvoa']
        row['total_dvoa_begining'] = numpy.nan
    else:
        row['blended_dvoa_begining'] = dvoa_new_df[(dvoa_new_df['team'] == row['team']) & (dvoa_new_df['season'] == row['season']) & (dvoa_new_df['week'] + 1 == row['week'])]['blended_dvoa'].iloc[0]
        row['total_dvoa_begining'] = dvoa_new_df[(dvoa_new_df['team'] == row['team']) & (dvoa_new_df['season'] == row['season']) & (dvoa_new_df['week'] + 1 == row['week'])]['total_dvoa'].iloc[0]
    return row


def apply_manual_fixes(row, manual_clean_dict):
    try:
        row['home_score'] = manual_clean_dict[row['game_id']]['home_score']
        row['away_score'] = manual_clean_dict[row['game_id']]['away_score']
    except:
        pass
    return row


def define_field_surfaces(game_df, surface_repl):
    ## copy frame ##
    temp_df = game_df.copy()
    ## remove neutrals ##
    temp_df = temp_df[
        (temp_df['neutral_field'] != 1) &
        (~pd.isnull(temp_df['home_score']))
    ].copy()
    ## standardize turf types ##
    temp_df['surface'] = temp_df['surface'].replace(surface_repl)
    ## generate df of field types by team ##
    fields_df = temp_df.groupby(
        ['home_team', 'season', 'surface']
    ).agg(
        games_played = ('home_score', 'count'),
    ).reset_index()
    ## get most played surface ##
    fields_df = fields_df.sort_values(
        by=['games_played'],
        ascending=[False]
    ).reset_index(drop=True)
    fields_df = fields_df.groupby(
        ['home_team', 'season']
    ).head(1)
    ## create new struc for handling start of season where team may not have a home game yet ##
    last_season = temp_df['season'].max()
    last_week = temp_df[
        temp_df['season'] == last_season
    ]['week'].max()
    curr_month = datetime.datetime.now().month
    ## if it's before week 1, increment current season ##
    if curr_month <= 9 and curr_month >= 4 and last_week > 1:
        last_season += 1
    else:
        pass
    ## new struc containing every team and season ##
    all_team_season_struc = []
    for season in range(temp_df['season'].min(), last_season + 1):
        for team in temp_df['home_team'].unique().tolist():
            all_team_season_struc.append({
                'team' : team,
                'season' : season
            })
    all_team_season_df = pd.DataFrame(all_team_season_struc)
    ## add fields ##
    all_team_season_df = pd.merge(
        all_team_season_df,
        fields_df[[
            'home_team', 'season', 'surface'
        ]].rename(columns={
            'home_team' : 'team'
        }),
        on=['team', 'season'],
        how='left'
    )
    ## fill missing ##
    all_team_season_df = all_team_season_df.sort_values(
        by=['team', 'season'],
        ascending=[True, True]
    ).reset_index(drop=True)
    all_team_season_df['surface'] = all_team_season_df.groupby(
        ['team']
    )['surface'].transform(lambda x: x.bfill().ffill())
    ## eliminate any possibility of duping on eventual merge w/ unique records ##
    all_team_season_df = all_team_season_df.drop_duplicates()
    return all_team_season_df


def define_time_advantages(game_df, timezones, timezone_overrides):
    ## helper to apply overrides ##
    def apply_tz_overrides(row, timezone_overrides):
        home_overide = None
        away_overide = None
        ## try to load overrides ##
        try:
            home_overide = timezone_overrides[row['home_team']]
        except:
            pass
        try:
            away_overide = timezone_overrides[row['away_team']]
        except:
            pass
        ## apply override if applicable ##
        ## home ##
        if home_overide is None:
            pass
        elif row['season'] <= home_overide['season']:
            row['home_tz'] = home_overide['tz_override']
        else:
            pass
        ## away ##
        if away_overide is None:
            pass
        elif row['season'] <= away_overide['season']:
            row['away_tz'] = away_overide['tz_override']
        else:
            pass
        return row
    ## copy frame ##
    temp_df = game_df.copy()
    peak_time = '14:00'
    ## add time zones ##
    temp_df['home_tz'] = temp_df['home_team'].replace(timezones).fillna('ET')
    temp_df['away_tz'] = temp_df['away_team'].replace(timezones).fillna('ET')
    ## apply overrides ##
    temp_df = temp_df.apply(
        apply_tz_overrides,
        timezone_overrides=timezone_overrides,
        axis=1
    )
    ## define optimals in ET ##
    temp_df['home_optimal_in_et'] = pd.Timestamp(peak_time)
    temp_df['away_optimal_in_et'] = pd.Timestamp(peak_time)
    ## home ##
    temp_df['home_optimal_in_et'] = numpy.where(
        temp_df['home_tz'] == 'ET',
        temp_df['home_optimal_in_et'].dt.time,
        numpy.where(
            temp_df['home_tz'] == 'CT',
            (temp_df['home_optimal_in_et'] + pd.Timedelta(hours=1)).dt.time,
            numpy.where(
                temp_df['home_tz'] == 'MT',
                (temp_df['home_optimal_in_et'] + pd.Timedelta(hours=2)).dt.time,
                numpy.where(
                    temp_df['home_tz'] == 'PT',
                    (temp_df['home_optimal_in_et'] + pd.Timedelta(hours=3)).dt.time,
                    temp_df['home_optimal_in_et'].dt.time
                )
            )
        )
    )
    ## away ##
    temp_df['away_optimal_in_et'] = numpy.where(
        temp_df['away_tz'] == 'ET',
        temp_df['away_optimal_in_et'].dt.time,
        numpy.where(
            temp_df['away_tz'] == 'CT',
            (temp_df['away_optimal_in_et'] + pd.Timedelta(hours=1)).dt.time,
            numpy.where(
                temp_df['away_tz'] == 'MT',
                (temp_df['away_optimal_in_et'] + pd.Timedelta(hours=2)).dt.time,
                numpy.where(
                    temp_df['away_tz'] == 'PT',
                    (temp_df['away_optimal_in_et'] + pd.Timedelta(hours=3)).dt.time,
                    temp_df['away_optimal_in_et'].dt.time
                )
            )
        )
    )
    ## get kickoff ##
    temp_df['gametimestamp'] = pd.to_datetime(temp_df['gametime'], format='%H:%M').dt.time
    ## define advantage ##
    temp_df['home_time_advantage'] = numpy.round(
        numpy.absolute(
            (
                pd.to_datetime(temp_df['gametimestamp'], format='%H:%M:%S') -
                pd.to_datetime(temp_df['away_optimal_in_et'], format='%H:%M:%S')
            ) / numpy.timedelta64(1, 'h')
        ) -
        numpy.absolute(
            (
                pd.to_datetime(temp_df['gametimestamp'], format='%H:%M:%S') -
                pd.to_datetime(temp_df['home_optimal_in_et'], format='%H:%M:%S')
            ) / numpy.timedelta64(1, 'h')
        )
    )
    return temp_df['home_time_advantage'].fillna(0)


## weather ##
## func to lookup weather from table ##
def get_weather(team, season, week, weather_df):
    if team == 'OAK' and season >= 2021:
        team = 'LV'
    elif team == 'LAC' and season < 2021:
        team = 'SD'
    elif team == 'LAR' and season < 2018:
        team = 'STL'
    else:
        pass
    temp = weather_df.loc[
        (weather_df['team'] == team) &
        (weather_df['week'] == week)
    ].iloc[0]['week_temp']
    return round(temp,1)


## func to apply weather to each row ##
def apply_weather(row, weather_df):
    home_temp = get_weather(
        row['home_team'],
        row['season'],
        row['week'],
        weather_df
    )
    away_temp = get_weather(
        row['away_team'],
        row['season'],
        row['week'],
        weather_df
    )
    row['home_temp'] = home_temp
    row['away_temp'] = away_temp
    return row



def game_data_merge():
    print('Merging game level datasets (fastr, market, pff, fbo)...')
    print('     Loading individual datasets...')
    game_df = pd.read_csv(
        '{0}/{1}'.format(
            package_dir,
            game_loc
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
    dvoa_projections_df = pd.read_csv(
        '{0}/{1}'.format(
            package_dir,
            dvoa_proj_loc
        ),
        index_col=0
    )
    dvoa_df = pd.read_csv(
        '{0}/{1}'.format(
            package_dir,
            dvoa_weekly_loc
        ),
        index_col=0
    )
    pff_df = pd.read_csv(
        '{0}/{1}'.format(
            package_dir,
            pff_loc
        ),
        index_col=0
    )
    weather_df = pd.read_csv(
        '{0}/{1}'.format(
            package_dir,
            weather_loc
        ),
        index_col=0
    )
    ## filter data and columns ##
    game_df = game_df[game_headers].rename(columns=game_repl).copy()
    spread_df = spread_df[[
        'game_id','home_line_open','home_line_close',
        'home_ats_pct', 'neutral_field', 'divisional_game', 'market_implied_elo_dif'
    ]].copy()
    dvoa_df = dvoa_df[['season','week','team','total_dvoa']].copy()
    dvoa_projections_df = dvoa_projections_df[
        dvoa_projections_df['season'] >= 2009
    ].copy()
    ## dvoa ##
    print('     Compiling FBO ratings...')
    dvoa_new_df = pd.merge(
        dvoa_projections_df,
        dvoa_df,
        on=['season','team'],
        how='left'
    )
    ## calc blended dvoa for in season total dvoa ##
    ## not actively used, but keeping for future use ##
    dvoa_new_df['blended_dvoa'] = numpy.where(
        dvoa_new_df['week'] <= 8,
        (
            (8-dvoa_new_df['week']) *
            dvoa_new_df['projected_total_dvoa'] +
            dvoa_new_df['week'] *
            dvoa_new_df['total_dvoa']
        ) / 8,
        dvoa_new_df['total_dvoa']
    )
    dvoa_new_df = dvoa_new_df.apply(calc_begining_dvoa, dvoa_new_df=dvoa_new_df.copy(), axis=1)
    ## combine it all ##
    print('     Joining datasets...')
    ## remove probowls ##
    game_df = game_df[
        game_df['home_team'].isin(
            list(pbp_team_standard_dict.values())
        )
    ].copy()
    ## merge data sets ##
    new_df = pd.merge(
        game_df,
        spread_df,
        on=['game_id'],
        how='left'
    )
    # merge dvoa info ##
    home_dvoa_merge = dvoa_new_df.rename(columns=home_dvoa_repl).copy()
    away_dvoa_merge = dvoa_new_df.rename(columns=away_dvoa_repl).copy()
    new_df = pd.merge(
        new_df,
        home_dvoa_merge,
        on=['season','week','home_team'],
        how='left'
    )
    new_df = pd.merge(
        new_df,
        away_dvoa_merge,
        on=['season','week','away_team'],
        how='left'
    )
    ## report potential join errors ##
    new_games = len(game_df) - len(new_df)
    print('          Joining spread and DVOA data changed number of games by {0}...'.format(new_games))
    null_df = new_df.copy()
    null_df = null_df[
        (pd.isnull(null_df['home_line_close'])) &
        (~pd.isnull(null_df['home_score']))
    ]
    print('          {0} games are missing spreads...'.format(
        len(null_df)
    ))
    ## add pff data and pff margins ##
    ## Load pff game grade data ##
    pff_df['pff_home_grade_delta'] = pff_df['home_overall_grade'] - pff_df['away_overall_grade']
    pff_df = pff_df[[
        'season','week','home_team','away_team','pff_home_grade_delta',
        'home_overall_grade','away_overall_grade'
    ]].copy()
    ## determine post v reg ##
    pff_df['type'] = numpy.where(
        pff_df['season'] <= 2020,
        numpy.where(
            pff_df['week'] > 17,
            'post',
            'reg'
        ),
        numpy.where(
            pff_df['week'] > 18,
            'post',
            'reg'
        )
    ).copy()
    ## drop week for join ##
    pff_df = pff_df[[
        'season','type','home_team','away_team','pff_home_grade_delta',
        'home_overall_grade','away_overall_grade'
    ]].copy()
    ## join pff to game file ##
    new_df = pd.merge(
        new_df,
        pff_df,
        on=['season','type','home_team','away_team'],
        how='left'
    )
    ## check for dupes ##
    new_games = len(new_df) - (len(game_df) + new_games)
    print('          Joining pff data changed number of games by {0}...'.format(
        new_games
    ))
    ## calc PFF margin from regression ##
    ## flat file ##
    pff_flat_df = pd.concat([
        ## home ##
        new_df.copy()[[
            'season', 'home_overall_grade', 'home_score', 'away_score'
        ]].rename(columns={
            'home_overall_grade' : 'overall_grade',
            'home_score' : 'pf',
            'away_score' : 'pa',
        }),
        new_df.copy()[[
            'season', 'away_overall_grade', 'away_score', 'home_score'
        ]].rename(columns={
            'away_overall_grade' : 'overall_grade',
            'away_score' : 'pf',
            'home_score' : 'pa',
        })
    ])
    pff_flat_df = pff_flat_df[pff_flat_df['season'] >= 2009]
    pff_flat_df = pff_flat_df[~pd.isnull(pff_flat_df['overall_grade'])]
    pff_flat_df['intercept_constant'] = 1
    pff_flat_df['margin'] = pff_flat_df['pf'] - pff_flat_df['pa']
    model = sm.OLS(pff_flat_df['margin'], pff_flat_df[['overall_grade', 'intercept_constant']], hasconst=True).fit()
    pff_intercept = model.params.intercept_constant
    pff_slope = model.params.overall_grade
    print('          PFF point margin model updated...')
    print('               rsq: {0}...'.format(round(model.rsquared,3)))
    print('               slope: {0}...'.format(round(pff_slope,3)))
    print('               intercept: {0}...'.format(round(pff_intercept,3)))
    ## apply ##
    ## hard coded pff margins ##
    ## deciding to freeze these to keep model consistent when looking backwards ##
    pff_intercept = -87.728
    pff_slope = 1.263
    new_df['home_pff_point_margin'] = pff_intercept + pff_slope * new_df['home_overall_grade']
    new_df['away_pff_point_margin'] = pff_intercept + pff_slope * new_df['away_overall_grade']
    ## fill missing games with straight margin ##
    new_df['home_pff_point_margin'] = new_df['home_pff_point_margin'].combine_first(new_df['home_score'] - new_df['away_score'])
    new_df['away_pff_point_margin'] = new_df['away_pff_point_margin'].combine_first(new_df['away_score'] - new_df['home_score'])
    ## add meta for new hfa model ##
    print('     Applying meta for new HFA model...')
    final_len = len(new_df)
    ## field ##
    print('          Defining field surface types...')
    fields_df = define_field_surfaces(new_df, pbp_surface_repl)
    new_df = pd.merge(
        new_df,
        fields_df.rename(columns={
            'team' : 'home_team',
            'surface' : 'home_surface'
        }),
        on=['home_team', 'season'],
        how='left'
    )
    new_df = pd.merge(
        new_df,
        fields_df.rename(columns={
            'team' : 'away_team',
            'surface' : 'away_surface'
        }),
        on=['away_team', 'season'],
        how='left'
    )
    new_df['home_surface_advantage'] = numpy.where(
        (new_df['home_surface'] != new_df['away_surface']) &
        (new_df['surface'].replace(pbp_surface_repl) == new_df['home_surface']),
        1,
        0
    )
    final_len_post_fields = len(new_df)
    print('          Joining fields data changed number of games by {0}...'.format(
        final_len_post_fields - final_len
    ))
    ## timezone ##
    print('          Defining timezone advantages...')
    new_df['home_time_advantage'] = define_time_advantages(
        new_df, pbp_timezones, pbp_timezone_overrides
    )
    final_len_post_tz = len(new_df)
    print('          Joining timezone data changed number of games by {0}...'.format(
        final_len_post_tz - final_len_post_fields
    ))
    ## weather ##
    print('          Defining weather...')
    new_df = new_df.apply(
        apply_weather,
        weather_df=weather_df,
        axis=1
    )
    new_df['home_temp_advantage'] = numpy.absolute(
        new_df['home_temp'] -
        new_df['away_temp']
    )
    final_len_post_weather = len(new_df)
    print('          Joining weather changed number of games by {0}...'.format(
        final_len_post_weather - final_len_post_tz
    ))
    print('     Manually cleaning up bad scores...')
    new_df = new_df[final_headers]
    new_df = new_df.apply(apply_manual_fixes, manual_clean_dict=manual_clean_dict, axis=1)
    ## export ##
    new_df.to_csv(
        '{0}{1}/current_file.csv'.format(
            package_dir,
            output_folder
        )
    )
    print('     Done!')
