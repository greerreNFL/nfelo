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


starting_season = config['data_pulls']['nflfastR']['participant_starting_year']
part_url = config['data_pulls']['nflfastR']['participant_loc']

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
    print('Updating Participant data...')
    ## pull data ##
    all_dfs = []
    while current_season <= ending_season:
        print('     Downloading {0} participant data...'.format(current_season))
        temp = pd.read_csv(
            '{0}/pbp_participation_{1}.csv'.format(
                part_url,
                current_season
            ),
        )
        temp['season'] = current_season
        all_dfs.append(temp)
        current_season += 1
    ## combine ##
    print('     Saving...')
    all_seasons = pd.concat(all_dfs)
    ## save ##
    all_seasons.to_csv(
        '{0}/{1}/participants.csv'.format(
            package_dir,
            output_folder
        )
    )


## define function to run data pull ##
def pull_nflfastR_participants():
    data_pull(current_season, ending_season, package_dir, output_folder)
