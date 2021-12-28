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
excluded_logos = config['data_pulls']['nflfastR']['team_exclusions']
logo_url = config['data_pulls']['nflfastR']['logo_loc']


## create output sub path ##
output_folder = '/data_sources/legacy_pbp'


## func that does the actual pull ##
def data_pull():
    print('Updating nflfastR Logo data...')
    ## pull data ##
    print('     Downloading logos...')
    logo_df = pd.read_csv(logo_url)
    print('     Formating file...')
    ## drop old logos and format names ##
    logo_df = logo_df[
        ~numpy.isin(
            logo_df['team_abbr'],
            excluded_logos
        )
    ]
    ## drop index to get proper sequential order after exclusions ##
    logo_df = logo_df.reset_index(drop=True)
    ## rename teams ##
    logo_df['team'] = logo_df['team_abbr'].copy().replace(name_standardization_dict)
    ## export ##
    print('     Saving file...')
    logo_df.to_csv(
        '{0}/{1}/logos.csv'.format(
            package_dir,
            output_folder
        )
    )



## define function to run data pull ##
def pull_nflfastR_logo():
    data_pull()
