import pandas as pd
import numpy
import pathlib

from ..Data import DataLoader
from ..Model import Nfelo
from ..Performance import NfeloGrader
from ..Analytics import NfeloAnalytics
from ..Utilities import bet_size

class NfeloFormatter:
    '''
    Class that formats data to be used in downstream pipelines
    '''

    def __init__(self, data:DataLoader, model:Nfelo, graded:NfeloGrader) -> None:
        print('Formatting Output Files...')
        self.data = data
        self.model = model
        self.graded = graded
        self.output_loc = '{0}/Data/Formatted Data'.format(
            pathlib.Path(__file__).parent.parent.resolve()
        )
        self.external_folder = '{0}/output_data'.format(
            pathlib.Path(__file__).parent.parent.parent.resolve()
        )
        ## init an NfeloAnalytics instance, which handled
        ## loading, compiling and saving of the team and most_recent_team files
        self.analytics = NfeloAnalytics(self.model)
        ## format data ##
        self.gen_rolling_hfa()
        self.gen_scored_games()
        self.gen_wepa_flat()
        self.gen_wt_ratings()
        self.gen_projections()
        self.gen_most_recent_elo_file()
        self.gen_cur_w_analytics()
 
    def gen_rolling_hfa(self):
        '''
        Generate a rolling HFA file by season and week
        '''
        self.data.current_file[[
            'season', 'week', 'hfa_base'
        ]].groupby(['season', 'week']).agg(
            rolling_hfa = ('hfa_base', 'median')
        ).reset_index().to_csv(
            '{0}/rolling_hfa.csv'.format(self.output_loc)
        )
    
    def gen_scored_games(self):
        '''
        Formats the grader to produce a "scored individual games" file
        '''
        self.graded.graded_games[[
            'season', 'week', '538_brier', 'qbelo_brier', 'nfelo_close_brier',
            'nfelo_unregressed_brier', 'market_brier', 'market_open_brier',
            '538_open_ats', 'qbelo_open_ats', 'nfelo_open_ats',
            '538_ats', 'qbelo_ats', 'nfelo_close_ats',
            '538_open_ats_be', 'qbelo_open_ats_be', '538_ats_be',
            'qbelo_ats_be', 'nfelo_open_ats_be', 'nfelo_close_ats_be',
            '538_se', 'qbelo_se', 'nfelo_close_se', 'nfelo_unregressed_se',
            'market_se', 'market_open_se',
            '538_su', 'qbelo_su', 'nfelo_close_su', 'nfelo_unregressed_su',
            'market_su', 'market_open_su',
            'market_home_line', 'qbelo_home_line', 'nfelo_close_home_line',
            'nfelo_unregressed_ats_be'
        ]].rename(columns={
            'nfelo_close_brier' : 'nfelo_brier',
            '538_ats' : '538_close_ats',
            'qbelo_ats' : 'qbelo_close_ats',
            '538_ats' : '538_close_ats',
            '538_open_ats_be' : '538_open_ats_break_even',
            'qbelo_open_ats_be' : 'qbelo_open_ats_break_even',
            '538_ats_be' : '538_close_ats_break_even',
            'qbelo_ats_be' : 'qbelo_close_ats_break_even',
            'market_home_line' : 'home_line_close',
            'qbelo_home_line' : 'qbelo_home_line_close_rounded',
            'nfelo_close_home_line' : 'nfelo_home_line_close_rounded',
            'nfelo_unregressed_ats_be' : 'nfelo_close_ats_be_unregressed',
            'nfelo_close_se' : 'nfelo_se',
            'nfelo_close_su' : 'nfelo_su'
        }).to_csv(
            '{0}/scored_individual_games.csv'.format(
                self.output_loc
            )
        )

    def gen_wepa_flat(self):
        '''
        Generates a flattened wepa file
        '''
        ## copy data ##
        df = self.data.current_file.copy()
        ## flatten the file ##
        flat = pd.concat([
            ## home ##
            df[[
                'game_id', 'home_team', 'away_team', 'season', 'game_number_home',
                'home_margin', 'away_margin', 
                'home_offensive_epa', 'home_defensive_epa', 'home_net_epa', 'away_net_epa',
                'home_offensive_wepa', 'home_defensive_wepa', 'home_net_wepa',
                'away_offensive_wepa', 'away_defensive_wepa', 'away_net_wepa'
            ]].rename(columns={
                'home_team' : 'team',
                'away_team' : 'opponent',
                'game_number_home' : 'game_number',
                'home_margin' : 'margin',
                'away_margin' : 'margin_against',
                'home_offensive_epa' : 'epa',
                'home_defensive_epa' : 'epa_against',
                'home_net_epa' : 'net_epa',
                'away_net_epa' : 'net_epa_against',
                ## the naming in the original wepa flat file is bonkers
                ## tldr is the "d_wepa_against" being the teams actual defensive
                ## wepa is an artificat of a really lazy joining logic ##
                ## where the "defteam" is always "against", so when home is on d
                ## and therefore the "defteam" we want to join, its the "against"
                'home_offensive_wepa' : 'wepa',
                'home_defensive_wepa' : 'd_wepa_against',
                'home_net_wepa' : 'wepa_net',
                'away_offensive_wepa' : 'wepa_against',
                'away_defensive_wepa' : 'd_wepa',
                'away_net_wepa' : 'wepa_net_opponent'
            }),
            ## away ##
            df[[
                'game_id', 'away_team', 'home_team', 'season', 'game_number_away',
                'away_margin', 'home_margin', 
                'away_offensive_epa', 'away_defensive_epa', 'away_net_epa', 'home_net_epa',
                'away_offensive_wepa', 'away_defensive_wepa', 'away_net_wepa',
                'home_offensive_wepa', 'home_defensive_wepa', 'home_net_wepa'
            ]].rename(columns={
                'away_team' : 'team',
                'home_team' : 'opponent',
                'game_number_away' : 'game_number',
                'away_margin' : 'margin',
                'home_margin' : 'margin_against',
                'away_offensive_epa' : 'epa',
                'away_defensive_epa' : 'epa_against',
                'away_net_epa' : 'net_epa',
                'home_net_epa' : 'net_epa_against',
                ## the naming in the original wepa flat file is bonkers
                ## tldr is the "d_wepa_against" being the teams actual defensive
                ## wepa is an artificat of a really lazy joining logic ##
                ## where the "defteam" is always "against", so when home is on d
                ## and therefore the "defteam" we want to join, its the "against"
                'away_offensive_wepa' : 'wepa',
                'away_defensive_wepa' : 'd_wepa_against',
                'away_net_wepa' : 'wepa_net',
                'home_offensive_wepa' : 'wepa_against',
                'home_defensive_wepa' : 'd_wepa',
                'home_net_wepa' : 'wepa_net_opponent'
            })
        ])
        ## drop games with no epa ##
        flat = flat[~pd.isnull(flat['epa'])].copy()
        ## sort ##
        flat = flat.sort_values(
            by=['team', 'season', 'game_number'],
            ascending=[True, True, True]
        ).reset_index(drop=True)
        ## save ##
        flat.to_csv(
            '{0}/wepa_flat_file.csv'.format(self.output_loc)
        )
    
    def gen_wt_ratings(self):
        '''
        Format the win total ratings to match downstream format
        '''
        ## load wt ratings and team logos/colors, which need to be added ##
        wt = self.data.db['wt_ratings'].copy()
        logo = self.data.db['logos'].copy()
        ## merge ##
        wt = pd.merge(
            wt,
            logo[[
                'team_abbr', 'team_nick', 'team_color', 'team_color2',
                'team_logo_espn'
            ]].groupby(['team_abbr']).head(1).rename(columns={
                'team_abbr' : 'team'
            }),
            on=['team'],
            how='left'
        )
        ## reorder cols and save ##
        wt[[
            'team', 'season', 'wt_rating', 'wt_rating_elo',
            'sos', 'line', 'over_odds', 'under_odds',
            'line_adj', 'team_nick', 'team_color', 'team_color2', 'team_logo_espn',
            'over_probability', 'under_probability', 'hold'
        ]].to_csv(
            '{0}/wt_ratings.csv'.format(self.output_loc)
        )

    def gen_projections(self):
        '''
        Generates projections from the passed Nfelo model
        '''
        ## check if they exist ##
        if self.model.projections is None:
            return
        ## format ##
        proj = self.model.projections[[
            'game_id', 'season', 'week', 'away_team', 'home_team',
            'home_line_open', 'home_line_close', 'nfelo_home_line_close', 'nfelo_home_line_base', 'nfelo_spread_delta',
            'nfelo_home_probability_close', 'starting_nfelo_home', 'starting_nfelo_away', 'nfelo_dif_pre_adjustment', 'nfelo_dif_base',
            'market_regression_factor_close', 'nfelo_dif_close', 'market_elo_dif_close', 'hfa_base_mod', 'home_net_bye_mod',
            'div_game_mod', 'dif_surface_mod', 'home_time_advantage_mod', 'hfa_mod', 'home_net_qb_mod',
            'home_538_qb_adj', 'away_538_qb_adj', 'home_538_qb', 'away_538_qb', 'away_loss_prob_close',
            'away_push_prob_close', 'away_cover_prob_close', 'away_close_ev', 'home_loss_prob_close', 'home_push_prob_close',
            'home_cover_prob_close', 'home_close_ev', 'old_game_id', 'game_date', 'game_day',
            'gametime'
        ]].rename(columns={
            'nfelo_home_line_close' : 'home_closing_line_rounded_nfelo',
            'nfelo_home_line_base' : 'home_line_pre_regression',
            'nfelo_home_probability_close' : 'home_probability_nfelo',
            'starting_nfelo_home' : 'home_nfelo_elo',
            'starting_nfelo_away' : 'away_nfelo_elo',
            'nfelo_dif_pre_adjustment' : 'home_dif',
            'nfelo_dif_base' : 'home_dif_pre_reg',
            'market_regression_factor_close' : 'market_regression_factor',
            'nfelo_dif_close' : 'regressed_dif',
            'market_elo_dif_close' : 'market_implied_elo_dif',
            'hfa_base_mod' : 'base_hfa',
            'div_game_mod' : 'div_mod',
            'dif_surface_mod' : 'surface_mod',
            'home_time_advantage_mod' : 'time_mod',
            'hfa_mod' : 'home_net_HFA_mod',
            'away_loss_prob_close' : 'away_loss_prob',
            'away_push_prob_close' : 'away_push_prob',
            'away_cover_prob_close' : 'away_cover_prob',
            'away_close_ev' : 'away_ev',
            'home_loss_prob_close' : 'home_loss_prob',
            'home_push_prob_close' : 'home_push_prob',
            'home_cover_prob_close' : 'home_cover_prob',
            'home_close_ev' : 'home_ev',
            'game_date' : 'gameday',
            'game_day' : 'weekday'
        }).copy()
        ## add ev ##
        proj['sort'] = proj[['home_ev', 'away_ev']].max(axis=1)
        ## sort ##
        proj = proj.sort_values(by=['sort'],ascending=[False]).reset_index(drop=True)
        ## add bet size ##
        proj['bet_size'] = bet_size(
            proj['home_cover_prob'],
            proj['home_loss_prob']
        )
        ## save ##
        proj.to_csv(
            ## change to outptu folder once confirmed the formatter is working correctly ##
            '{0}/projected_spreads.csv'.format(self.external_folder)
        )
        ## add to historics ##
        hist = pd.read_csv(
            '{0}/historic_projected_spreads.csv'.format(self.external_folder),
            index_col=0
        )
        ## remove anything with matching game id ##
        hist = hist[
            ~numpy.isin(
                hist['game_id'],
                proj['game_id'].unique().tolist()
            )
        ].copy()
        ## append ##
        hist = pd.concat([hist,proj]).reset_index(drop=True)
        ## save ##
        hist.to_csv(
            ## change to outptu folder once confirmed the formatter is working correctly ##
            '{0}/historic_projected_spreads.csv'.format(self.external_folder)
        )
        ## create a flat file of current elos ##
        flat = pd.concat([
            proj[['season', 'week', 'home_team', 'home_nfelo_elo', 'home_538_qb_adj']].rename(columns={
                'home_team' : 'team',
                'home_nfelo_elo' : 'nfelo_base',
                'home_538_qb_adj' : 'qb_adj'
            }),
            proj[['season', 'week', 'away_team', 'away_nfelo_elo', 'away_538_qb_adj']].rename(columns={
                'away_team' : 'team',
                'away_nfelo_elo' : 'nfelo_base',
                'away_538_qb_adj' : 'qb_adj'
            })
        ])
        ## calc the all in nfelo ##
        flat['nfelo'] = flat['nfelo_base'] + flat['qb_adj']
        ## convert to spread ##
        flat['pts_vs_avg'] = (flat['nfelo'] - flat['nfelo'].median()) / 25
        ## save ##
        try:
            existing = pd.read_csv(
                '{0}/elo_snapshot.csv'.format(self.external_folder),
                index_col=0
            )
            ## merge ##
            existing = existing[~numpy.isin(
                existing['team'],
                flat['team'].unique().tolist()
            )].copy()
            if len(existing) == 0:
                flat.sort_values(
                    by=['nfelo'],
                    ascending=[False]
                ).reset_index(drop=True).to_csv(
                    '{0}/elo_snapshot.csv'.format(self.external_folder)
                )
            else:
                pd.concat([
                    flat,existing
                ]).sort_values(
                    by=['nfelo'],
                    ascending=[False]
                ).reset_index(drop=True).to_csv(
                    '{0}/elo_snapshot.csv'.format(self.external_folder)
                )
        except:
            flat.sort_values(
                by=['nfelo'],
                ascending=[False]
            ).reset_index(drop=True).to_csv(
                '{0}/elo_snapshot.csv'.format(self.external_folder)
            )


    
    def gen_most_recent_elo_file(self):
        '''
        Generates the most recent elo file
        '''
        df = self.model.updated_file.copy()
        ## add adjs to the nfelos ##
        df['nfelo_home'] = df['ending_nfelo_home'] + df['home_538_qb_adj']
        df['nfelo_starting_home'] = df['starting_nfelo_home'] + df['home_538_qb_adj']
        df['nfelo_away'] = df['ending_nfelo_away'] + df['away_538_qb_adj']
        df['nfelo_starting_away'] = df['starting_nfelo_away'] + df['away_538_qb_adj']
        ## flatten ##
        flat = pd.concat([
            ## home ##
            df[[
                'home_team', 'game_id', 'season', 'week',
                'nfelo_home', 'nfelo_starting_home', 'home_538_qb_adj'
            ]].rename(columns={
                'home_team' : 'team',
                'nfelo_home' : 'nfelo',
                'nfelo_starting_home' : 'nfelo_starting',
                'home_538_qb_adj' : '538_qb_adj'
            }),
            ## away ##
            df[[
                'away_team', 'game_id', 'season', 'week',
                'nfelo_away', 'nfelo_starting_away', 'away_538_qb_adj'
            ]].rename(columns={
                'away_team' : 'team',
                'nfelo_away' : 'nfelo',
                'nfelo_starting_away' : 'nfelo_starting',
                'away_538_qb_adj' : '538_qb_adj'
            })
        ])
        ## re-sort ##
        flat = flat.sort_values(
            by=['team', 'season', 'week'],
            ascending=[True, True, True]
        ).reset_index(drop=True)
        ## create lags ##
        flat['nfelo_t_1'] = flat.groupby(['team'])['nfelo'].shift(1)
        flat['nfelo_soy'] = flat.groupby(['team', 'season'])['nfelo_starting'].transform('first')
        ## add deltas ##
        flat['nfelo_wow_delta'] = flat['nfelo'] - flat['nfelo_t_1']
        flat['nfelo_ytd_delta'] = flat['nfelo'] - flat['nfelo_soy']
        ## get most recent record of df ##
        flat = flat.groupby(['team']).tail(1).reset_index(drop=True)
        ## save ##
        flat.to_csv(
            '{0}/most_recent_elo_file.csv'.format(
                self.output_loc
            )
        )
    
    def gen_cur_w_analytics(self):
        '''
        Replicates the current_file_w_analytics
        This file has a lot of extraneous fields, so this formatted focuses mainly
        on ensuring the fields taht are used down stream are there and named correctly.
        It is not a faithfully full representatin of the original
        '''
        df = self.model.updated_file[[
            'game_id', 'type', 'season', 'week', 'home_team',
            'away_team', 'home_score', 'away_score', 'home_margin', 'home_line_open',
            'home_line_close', 'ats_pct', 'game_date', 'game_day', 'stadium',
            'stadium_id', 'roof', 'surface', 'temperature', 'wind',
            'div_game', 'is_neutral', 'home_projected_dvoa', 'away_projected_dvoa', 'starting_nfelo_home',
            'starting_nfelo_away', 'ending_nfelo_home', 'ending_nfelo_away', 'nfelo_home_probability_close',
            'nfelo_home_probability_open', 'nfelo_home_line_close',
            'home_net_wepa', 'away_net_wepa', 'home_net_wepa_point_margin', 'away_net_wepa_point_margin', 'home_pff_point_margin',
            'away_pff_point_margin', 'home_538_qb_adj', 'away_538_qb_adj', 'home_close_ev', 'away_close_ev',
            'home_open_ev', 'away_open_ev', 'hfa_mod', 'home_bye_mod', 'away_bye_mod',
            'div_game_mod', 'dif_surface_mod', 'home_time_advantage_mod', 'home_time_advantage', 'away_moneyline',
            'home_moneyline', 'away_spread_odds', 'home_spread_odds', 'nfelo_home_line_base', 'qbelo_prob1',
            'nfelo_home_probability_base', '538_home_line_close', 'qbelo_home_line_close',
        ]].copy().rename(columns={
            'ats_pct' : 'home_ats_pct',
            'div_game' : 'divisional_game',
            'is_neutral' : 'neutral_field',
            'nfelo_home_probability_close' : 'nfelo_home_probability',
            'home_close_ev' : 'home_ev',
            'away_close_ev' : 'away_ev',
            'home_open_ev' : 'home_ev_open',
            'away_open_ev' : 'away_ev_open',
            'div_game_mod' : 'div_mod',
            'dif_surface_mod' : 'surface_mod',
            'home_time_advantage_mod' : 'time_mod',
            'nfelo_home_line_base' : 'nfelo_home_line_close_pre_regression',
            'nfelo_home_probability_base' : 'nfelo_home_probability_pre_regression',
        })
        ## save ##
        df.to_csv(
            '{0}/current_file_with_analytics.csv'.format(
                self.output_loc
            )
        )
