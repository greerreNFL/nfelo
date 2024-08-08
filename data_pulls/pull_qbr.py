## packages ##
import pandas as pd
import numpy
import pathlib
import json


## get package directory ##
package_dir = pathlib.Path(__file__).parent.parent.resolve()

## load config file ##
config = None
with open('{0}/config.json'.format(package_dir), 'r') as fp:
    config = json.load(fp)

url = config['data_pulls']['qbr']['url']


## create output sub path ##
output_folder = '/data_sources/qbr'


def pull_qbr():
    print('Pulling QBR...')
    try:
        df = pd.read_csv(url)
        df.to_csv('{0}{1}/seasonal_qbr.csv'.format(
            package_dir, output_folder
        ))
        print('     Success!')
    except Exception as e:
        print('     Pull failed:')
        print('          {0}'.format(e))
