import pandas as pd
import numpy

from ..Utilities import (
    brier_score, grade_bet_vector, grade_su_vector,
    market_correl, adj_brier, grade_se_vector,
    ats_adj_brier
)

class NfeloGraderModel:
    '''
    Defines the model in the context of grading -- ie how to look up
    its projections in the table and return necessary scoring metrics
    
    '''
    def __init__(self, df:pd.DataFrame,
        model_name:str, model_line_col:str, market_line_col:str,
        model_probability_col:str, home_ev_col:str, away_ev_col:str
    ):
        self.df = df.copy()
        self.model_name = model_name
        self.model_line = self.df[model_line_col]
        self.market_line = self.df[market_line_col]
        self.win_prob = self.df[model_probability_col]
        self.result = self.df['home_margin']
        self.home_ev = None if home_ev_col is None else self.df[home_ev_col]
        self.away_ev = None if away_ev_col is None else self.df[away_ev_col]
        ## add scores ##
        self.add_brier()
        self.add_line()
        self.add_se()
        self.add_ats()
        self.add_su()
        self.score_record = self.gen_score_record()
    
    def add_brier(self):
        '''
        Adds the brier score
        '''
        self.df['{0}_brier'.format(self.model_name)] = brier_score(
            self.win_prob, self.result
        )
    
    def add_line(self):
        '''
        Adds the model line
        '''
        self.df['{0}_home_line'.format(self.model_name)] = self.model_line

    def add_se(self):
        '''
        Adds the squared error of the model 
        '''
        self.df['{0}_se'.format(self.model_name)] = (
            self.model_line + self.result
        ) ** 2
    
    def add_ats(self):
        '''
        adds the ats of the model
        '''
        self.df['{0}_ats'.format(self.model_name)] = grade_bet_vector(
            self.model_line, self.market_line, self.result,
            self.home_ev, self.away_ev, False
        )
        self.df['{0}_ats_be'.format(self.model_name)] = grade_bet_vector(
            self.model_line, self.market_line, self.result,
            self.home_ev, self.away_ev, True
        )

    def add_su(self):
        '''
        Adds the straight up results of the model
        '''
        self.df['{0}_su'.format(self.model_name)] = grade_su_vector(
            self.model_line, self.result
        )
    
    def merge_with(self, df):
        '''
        Adds a filtered version of itself to a target data frame
        '''
        df = pd.merge(
            df,
            self.df[[
                'game_id', '{0}_home_line'.format(self.model_name), '{0}_brier'.format(self.model_name),
                '{0}_se'.format(self.model_name), '{0}_ats'.format(self.model_name),
                '{0}_ats_be'.format(self.model_name),'{0}_su'.format(self.model_name)
            ]],
            on=['game_id'],
            how='left'
        )
        ## return the merged df ##
        return df

    def gen_score_record(self):
        '''
        Generates a score record that can be combined with others in a df
        '''
        return {
            'model_name' : self.model_name,
            'home_line' : self.model_line,
            'brier' : self.df['{0}_brier'.format(self.model_name)].sum(),
            'brier_per_game' : self.df['{0}_brier'.format(self.model_name)].mean(),
            'su' : self.df['{0}_su'.format(self.model_name)].mean(),
            'ats' : self.df['{0}_ats'.format(self.model_name)].mean(),
            'ats_be' : self.df['{0}_ats_be'.format(self.model_name)].mean(),
            'ats_be_play_pct' : (
                self.df['{0}_ats_be'.format(self.model_name)].count() /
                self.df['{0}_ats'.format(self.model_name)].fillna(0).count()
            ),
            'market_correl' : market_correl(
                self.model_line, self.market_line
            ),
            'brier_adj' : adj_brier(
                self.df['{0}_brier'.format(self.model_name)].sum(),
                market_correl(self.model_line, self.market_line)
            ),
            'brier_ats_adj' : ats_adj_brier(
                self.df['{0}_brier'.format(self.model_name)].sum(),
                self.df['{0}_ats_be'.format(self.model_name)].mean(),
                (
                    self.df['{0}_ats_be'.format(self.model_name)].count() /
                    self.df['{0}_ats'.format(self.model_name)].fillna(0).count()
                )
            ),
            'se' : self.df['{0}_se'.format(self.model_name)].mean()
        }

    def print_output(self):
        '''
        Prints the models performance across different measures
        '''
        print('     {0} Results:'.format(self.model_line))
        print('          Brier......{0}'.format(
            self.df['{0}_brier'.format(self.model_name)].sum()
        ))
        print('          SU.........{0}'.format(
            self.df['{0}_su'.format(self.model_name)].mean()
        ))
        print('          ATS........{0}'.format(
            self.df['{0}_ats'.format(self.model_name)].mean()
        ))
        print('          ATS BE.....{0}'.format(
            self.df['{0}_ats_be'.format(self.model_name)].mean()
        ))