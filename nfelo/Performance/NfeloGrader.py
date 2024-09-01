import pandas as pd
import numpy

import pathlib

from .NfeloGraderModel import NfeloGraderModel

class NfeloGrader:
    '''
    Takes an updated model DF and scores each model
    '''
    ## default models ##
    models = {
        'nfelo_unregressed' : {
            'model_line' : 'nfelo_home_line_base',
            'market_line' : 'home_line_close',
            'model_prob' : 'nfelo_home_probability_base',
            'home_ev' : None,
            'away_ev' : None
        },
        'nfelo_open' : {
            'model_line' : 'nfelo_home_line_open',
            'market_line' : 'home_line_open',
            'model_prob' : 'nfelo_home_probability_open',
            'home_ev' : 'home_open_ev',
            'away_ev' : 'away_open_ev'
        },
        'nfelo_close' : {
            'model_line' : 'nfelo_home_line_close',
            'market_line' : 'home_line_close',
            'model_prob' : 'nfelo_home_probability_close',
            'home_ev' : 'home_close_ev',
            'away_ev' : 'away_close_ev'
        },
        'market' : {
            'model_line' : 'home_line_close',
            'market_line' : 'home_line_close',
            'model_prob' : 'market_home_probability_close',
            'home_ev' : None,
            'away_ev' : None
        },
        'market_open' : {
            'model_line' : 'home_line_open',
            'market_line' : 'home_line_open',
            'model_prob' : 'market_home_probability_open',
            'home_ev' : None,
            'away_ev' : None
        },
        '538' : {
            'model_line' : '538_home_line_close',
            'market_line' : 'home_line_close',
            'model_prob' : 'elo_prob1',
            'home_ev' : None,
            'away_ev' : None
        },
        '538_open' : {
            'model_line' : '538_home_line_close',
            'market_line' : 'home_line_open',
            'model_prob' : 'elo_prob1',
            'home_ev' : None,
            'away_ev' : None
        },
        'qbelo' : {
            'model_line' : 'qbelo_home_line_close',
            'market_line' : 'home_line_close',
            'model_prob' : 'qbelo_prob1',
            'home_ev' : None,
            'away_ev' : None
        },
        'qbelo_open' : {
            'model_line' : 'qbelo_home_line_close',
            'market_line' : 'home_line_open',
            'model_prob' : 'qbelo_prob1',
            'home_ev' : None,
            'away_ev' : None
        }
    }

    def __init__(self, df:pd.DataFrame):
        self.df = df.copy()
        self.graded_games = df[['game_id', 'season', 'week']].copy()
        self.graded_records = []
        self.grade_models()

    def grade_models(self):
        '''
        Initialized and grades each model
        '''
        print('Grading Models...')
        for k,v in self.models.items():
            ## create model ##
            model = NfeloGraderModel(
                df=self.df,
                model_name=k,
                model_line_col=v['model_line'],
                market_line_col=v['market_line'],
                model_probability_col=v['model_prob'],
                home_ev_col=v['home_ev'],
                away_ev_col=v['away_ev']
            )
            ## append the record ##
            self.graded_records.append(model.score_record)
            ## add to graded games ##
            self.graded_games = model.merge_with(self.graded_games)
    
    def print_scores(self):
        '''
        Prints the graded df
        '''
        graded_summary = pd.DataFrame(self.graded_records)
        graded_summary = graded_summary.sort_values(
            by=['brier'],
            ascending=[False]
        ).reset_index(drop=True)
        print(graded_summary)
    
    def save_scores(self, loc=None):
        '''
        Saves the individual scores
        '''
        if loc is None:
            loc = '{0}/Data/Intermediate Data/scored_individual_games.csv'.format(
                pathlib.Path(__file__).parent.parent.resolve()
            )
        self.graded_games.to_csv(loc)