import pandas as pd
import numpy
import pathlib

from ..Utilities import merge_check
from ..Model import Nfelo

class NfeloAnalytics():
    '''
    The previous build of nfelo included team analytics that are not needed, but still expected
    by downstream pipelines.

    The NfeloAnalytics class adds these analytics to the current file produced by the 
    '''
    def __init__(self, nfelo:Nfelo):
        self.nfelo = nfelo
        self.team_file = self.compile()
        self.save_team_file()
      
    def flatten(self):
        '''
        Creates a flattened team df of relevant metrics
        '''
        df = self.nfelo.updated_file.copy()
        ## not used, but need a filler ##
        df['offensive_plays'] = 500
        df['defensive_plays'] = 500
        df['total_dvoa'] = numpy.nan
        df['blended_dvoa'] = numpy.nan
        ## flatten the file ##
        flat = pd.concat([
            ## home ##
            df[[
                'home_team', 'all_time_game_number_home', 'game_number_home', 'season', 'week',
                'game_id', 'away_team', 'home_score', 'away_score', 'home_margin', 'offensive_plays',
                'defensive_plays', 'home_offensive_epa', 'home_defensive_epa', 'home_net_epa',
                'home_offensive_wepa', 'home_defensive_wepa', 'home_net_wepa',
                'home_net_wepa_point_margin', 'home_pff_point_margin', 'total_dvoa', 'blended_dvoa'
            ]].rename(columns={
                'home_team' : 'team',
                'all_time_game_number_home' : 'all_time_game_number',
                'game_number_home' : 'season_game_number',
                'away_team' : 'opponent',
                'home_score' : 'points_for',
                'away_score' : 'points_against',
                'home_margin' : 'margin',
                'home_offensive_epa' : 'offensive_epa',
                'home_defensive_epa' : 'defensive_epa',
                'home_net_epa' : 'net_epa',
                'home_offensive_wepa' : 'offensive_wepa',
                'home_defensive_wepa' : 'defensive_wepa',
                'home_net_wepa' : 'net_wepa',
                'home_net_wepa_point_margin' : 'wepa_margin',
                'home_pff_point_margin' : 'pff_point_margin'
            }),
            ## away ##
            df[[
                'away_team', 'all_time_game_number_away', 'game_number_away', 'season', 'week',
                'game_id', 'home_team', 'away_score', 'home_score', 'away_margin', 'offensive_plays',
                'defensive_plays', 'away_offensive_epa', 'away_defensive_epa', 'away_net_epa',
                'away_offensive_wepa', 'away_defensive_wepa', 'away_net_wepa',
                'away_net_wepa_point_margin', 'away_pff_point_margin', 'total_dvoa', 'blended_dvoa'
            ]].rename(columns={
                'away_team' : 'team',
                'all_time_game_number_away' : 'all_time_game_number',
                'game_number_away' : 'season_game_number',
                'home_team' : 'opponent',
                'away_score' : 'points_for',
                'home_score' : 'points_against',
                'away_margin' : 'margin',
                'away_offensive_epa' : 'offensive_epa',
                'away_defensive_epa' : 'defensive_epa',
                'away_net_epa' : 'net_epa',
                'away_offensive_wepa' : 'offensive_wepa',
                'away_defensive_wepa' : 'defensive_wepa',
                'away_net_wepa' : 'net_wepa',
                'away_net_wepa_point_margin' : 'wepa_margin',
                'away_pff_point_margin' : 'pff_point_margin'
            })
        ])
        ## sort ##
        flat = flat.sort_values(
            by=['team', 'season', 'week'],
            ascending=[True, True, True]
        ).reset_index(drop=True)
        ## reset all time game counter to be from 2009 ##
        flat['all_time_game_number'] = flat.groupby(['team'])['season_game_number'].cumcount()+1
        ## return ##
        return flat
    
    def add_rolls(self, df):
        '''
        Translates select columns into rolling figures
        '''
        for metric in ['margin', 'wepa_margin', 'pff_point_margin']:
            ## ytd ##
            df['{0}_ytd'.format(metric)] = df.groupby(['team', 'season'])[metric].transform(lambda x: x.cumsum())
            ## rolling ##
            for roll in [8, 16]:
                df['{0}_L{1}'.format(metric, roll)] = df.groupby(['team', 'season'])[metric].transform(
                    lambda x: x.rolling(roll, min_periods=1).sum()
                )
        ## return ##
        return df

    def add_nfelo(self, df):
        '''
        Adds the ending nfelo
        '''
        ## get the ending nfelo in a df ##
        nfelo_df = pd.DataFrame(self.nfelo.elo_records)
        ## merge ##
        df = pd.merge(
            df,
            nfelo_df[[
                'team', 'game_id', 'starting_nfelo', 'ending_nfelo'
            ]].rename(columns={
                'starting_nfelo' : 'nfelo_start',
                'ending_nfelo' : 'nfelo'
            }),
            on=['team', 'game_id'],
            how='left'
        )
        ## return ##
        return df

    def compile(self):
        '''
        Wrapper that compiles the data into a team file
        '''
        ## get flat ##
        flat = self.flatten()
        flat = merge_check(self.add_rolls, flat, 'rolling analytics')
        flat = merge_check(self.add_nfelo, flat, 'starting and ending nfelo')
        ## order final columns ##
        flat = flat[[
            'team',
            'all_time_game_number',
            'season_game_number',
            'season',
            'week',
            'game_id',
            'opponent',
            'points_for',
            'points_against',
            'margin',
            'offensive_plays',
            'defensive_plays',
            'offensive_epa',
            'defensive_epa',
            'net_epa',
            'offensive_wepa',
            'defensive_wepa',
            'net_wepa',
            'wepa_margin',
            'pff_point_margin',
            'margin_ytd',
            'wepa_margin_ytd',
            'pff_point_margin_ytd',
            'total_dvoa',
            'blended_dvoa',
            'margin_L8',
            'wepa_margin_L8',
            'pff_point_margin_L8',
            'margin_L16',
            'wepa_margin_L16',
            'pff_point_margin_L16',
            ## '538_elo',
            'nfelo',
            'nfelo_start'
        ]].copy()
        ## return ##
        return flat
    
    def save_team_file(self):
        '''
        Save team file to intermediate data
        '''
        ## output location ##
        loc = pathlib.Path(__file__).parent.parent.resolve()
        ## full ##
        self.team_file.to_csv(
            '{0}/Data/Formatted Data/team_file.csv'.format(loc)
        )
        ## most recent ##
        self.team_file.groupby(['team']).tail(1).reset_index(drop=True).to_csv(
            '{0}/Data/Formatted Data/most_recent_team_file.csv'.format(loc)
        )
    