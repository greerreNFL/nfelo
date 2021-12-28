## packages ##
import pandas as pd
import numpy
import datetime
import pathlib
import json


## get package directory ##
package_dir = pathlib.Path(__file__).parent.parent.resolve()

## load config file ##
config = None
with open('{0}/config.json'.format(package_dir), 'r') as fp:
    config = json.load(fp)


team_rename = config['data_pulls']['538']['team_rename']
data_url = config['data_pulls']['538']['game_loc']
starting_year = config['data_pulls']['538']['starting_year']
playoff_map = config['data_pulls']['538']['playoff_map']
column_rename_dict = config['data_pulls']['538']['column_rename_dict']

pbp_loc = '{0}/data_sources/legacy_pbp/games.csv'.format(
    package_dir
)

## create output sub path ##
output_folder = '/data_sources/538'

## helper to fix issues where neautral field swaps teams ##
def fix_neutrals(row):
    home_temp = row['home_team']
    away_temp = row['away_team']
    if row['neutral'] == 1 and pd.isnull(row['game_id']):
        row['home_team'] = away_temp
        row['away_team'] = home_temp
    else:
        pass
    return row


## func that does the actual pull ##
def data_pull():
    print('Updating 538 Game data...')
    ## pull data ##
    print('     Downloading data...')
    data_df = pd.read_csv(data_url)
    print('     Formating file...')
    print('          Dropping old seasons...')
    data_df = data_df[data_df['season'] >= starting_year]
    print('          Converting playoffs to booleans...')
    data_df['playoff'] = data_df['playoff'].replace(
        playoff_map
    )
    data_df['playoff'] = numpy.where(numpy.isnan(data_df['playoff']),'reg','post')
    print('          Renaming teams...')
    data_df['team1'] = data_df['team1'].replace(team_rename)
    data_df['team2'] = data_df['team2'].replace(team_rename)
    print('          Renaming columns...')
    data_df = data_df.rename(columns=column_rename_dict)
    print('     Adding nflfastR ids...')
    scraper_df = pd.read_csv(pbp_loc)[[
        'home_team',
        'away_team',
        'season',
        'type',
        'game_id',
    ]]
    ## merge nflfastR ##
    merged_df = pd.merge(
        data_df,
        scraper_df,
        on=['home_team','away_team','season','type'],
        how='left'
    )
    ## check for missing ids ##
    merged_df = merged_df.apply(fix_neutrals, axis=1)
    merged_df = merged_df.drop(columns=['game_id'])
    ## remerging ##
    merged_df = pd.merge(
        merged_df,
        scraper_df,
        on=['home_team','away_team','season','type'],
        how='left'
    )
    missing_ids = len(merged_df[
        pd.isnull(merged_df['game_id'])
    ])
    if missing_ids > 0:
        print('          ** {0} games do not have ids!!! **'.format(
            missing_ids
        ))
    else:
        pass
    print('     Saving file...')
    merged_df.to_csv(
        '{0}{1}/538_game_data.csv'.format(
            package_dir,
            output_folder
        )
    )



## define function to run data pull ##
def pull_538_games():
    data_pull()
