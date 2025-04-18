import pandas as pd
import pathlib
import json

from .Data import DataLoader
from .Model import Nfelo
from .Performance import NfeloGrader
from .Formatting import NfeloFormatter

def update_nfelo():
    '''
    Updates the nfelo model and saves the current file
    '''
    ## load config ##
    config_loc = '{0}/config.json'.format(
        pathlib.Path(__file__).parent.parent.resolve()
    )
    with open(config_loc, 'r') as fp:
        config = json.load(fp)
    ## load data ##
    data = DataLoader()
    nfelo = Nfelo(
        data=data,
        config=config['models']['nfelo']
    )
    nfelo.run()
    nfelo.save_reversions()
    ## save some output ##
    nfelo.updated_file.to_csv(
        '{0}/Data/Intermediate Data/current_file_w_nfelo.csv'.format(
            pathlib.Path(__file__).parent.resolve()
        )
    )
    ## grade #
    graded = NfeloGrader(nfelo.updated_file)
    ## format ##
    nfelo.project_spreads()
    formatting = NfeloFormatter(
        data=data, model=nfelo, graded=graded
    )