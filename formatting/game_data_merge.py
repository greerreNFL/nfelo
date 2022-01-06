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

pbp_team_standard_dict = config['data_pulls']['nflfastR']['team_standardization']
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
    print('               slope: {0}...'.format(round(pff_intercept,3)))
    ## apply ##
    ## hard coded pff margins ##
    ## deciding to freeze these to keep model consistent when looking backwards ##
    pff_intercept = 1.263
    pff_slope = -87.728
    new_df['home_pff_point_margin'] = pff_intercept + pff_slope * new_df['home_overall_grade']
    new_df['away_pff_point_margin'] = pff_intercept + pff_slope * new_df['away_overall_grade']
    ## fill missing games with straight margin ##
    new_df['home_pff_point_margin'] = new_df['home_pff_point_margin'].combine_first(new_df['home_score'] - new_df['away_score'])
    new_df['away_pff_point_margin'] = new_df['away_pff_point_margin'].combine_first(new_df['away_score'] - new_df['home_score'])
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
