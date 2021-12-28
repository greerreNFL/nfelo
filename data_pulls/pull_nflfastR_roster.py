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
starting_season = config['data_pulls']['nflfastR']['pbp_starting_year']
roster_url = config['data_pulls']['nflfastR']['roster_loc']


## create output sub path ##
output_folder = '/data_sources/legacy_pbp'

## establish range for downloads ##
current_season = starting_season
ending_season = None

## set max season ##
if datetime.date.today() > datetime.date(
    year=datetime.date.today().year,
    month=9,
    day=6
):
    ending_season = datetime.date.today().year
else:
    ending_season = datetime.date.today().year - 1


def data_pull(current_season, ending_season, package_dir, output_folder):
    ## pull data ##
    print('Updating Roster data...')
    ## pull data ##
    all_dfs = []
    while current_season <= ending_season:
        print('     Downloading {0} roster data...'.format(current_season))
        roster_df = pd.read_csv(
            '{0}/roster_{1}.csv?raw=true'.format(
                roster_url,
                current_season
            ),
            low_memory=False
        )
        all_dfs.append(roster_df)
        current_season += 1
    ## combine ##
    print('     Combining and formatting...')
    all_seasons = pd.concat(all_dfs)
    all_seasons['team'] = all_seasons['team'].replace(
        name_standardization_dict
    )
    ## save ##
    print('     Saving...')
    all_seasons.to_csv(
        '{0}/{1}/roster.csv'.format(
            package_dir,
            output_folder
        )
    )


## define function to run data pull ##
def pull_nflfastR_roster():
    data_pull(current_season, ending_season, package_dir, output_folder)
