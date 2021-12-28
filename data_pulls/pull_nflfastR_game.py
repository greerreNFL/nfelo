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

name_standardization_dict = config['data_pulls']['nflfastR']['team_standardization']
game_url = config['data_pulls']['nflfastR']['game_loc']


## create output sub path ##
output_folder = '/data_sources/legacy_pbp'


## func that does the actual pull ##
def data_pull():
    print('Updating nflfastR Game data...')
    ## pull data ##
    print('     Downloading games...')
    game_df = pd.read_csv(game_url)
    print('     Formating...')
    ## rename teams ##
    game_df['home_team'] = game_df['home_team'].replace(name_standardization_dict)
    game_df['away_team'] = game_df['away_team'].replace(name_standardization_dict)
    ## replace game_id using standardized franchise names ##
    game_df['game_id'] = (
        game_df['season'].astype('str') +
        '_' +
        game_df['week'].astype('str').str.zfill(2) +
        '_' +
        game_df['away_team'] +
        '_' +
        game_df['home_team']
    )
    ## rename game_type to match traditional format ##
    game_df['type'] = game_df['game_type'].replace({
        'REG' : 'reg',
        'WC' : 'post',
        'DIV' : 'post',
        'CON' : 'post',
        'SB' : 'post',
    })
    ## export ##
    game_df.to_csv(
        '{0}/{1}/games.csv'.format(
            package_dir,
            output_folder
        )
    )



## define function to run data pull ##
def pull_nflfastR_game():
    data_pull()
