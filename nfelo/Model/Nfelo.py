import pandas as pd
import numpy
import statistics
import pathlib

from ..Data import DataLoader
from ..Utilities import (
    offseason_regression, probability_to_spread, elo_to_prob,
    regress_to_market, prob_to_elo, calc_cover_probs,
    calc_weighted_shift, calc_clv
)

class Nfelo:
    '''
    Primary elo model. 
    '''

    def __init__(self, data:DataLoader, config:dict):
        self.data = data
        self.config = config['nfelo_config']
        self.initial_elos = config['begining_elo']
        self.first_season = 2009
        self.current_file = data.current_file[
            data.current_file['season'] >= self.first_season
        ].copy()
        self.teams = self.current_file['home_team'].unique().tolist()
        self.current_elos = self.init_elos()
        self.yearly_elos = {}
        self.reversion_records = []
        self.elo_records = []
        self.updated_file = None
        self.updated_file_ext = None
        self.projections = None
    
    def init_elos(self):
        '''
        Initiallizes a dictionary that is used
        to look up elo values
        
        The structure always represents a snapshot of the team
        through their most recent game. Since on init there is no recent
        game, we use "week 0"
        '''
        current_elos = {}
        for team in self.teams:
            current_elos[team] = {
                'team' : team,
                'season' : self.first_season,
                'week' : 0,
                'game_id' : numpy.nan,
                'opponent' : numpy.nan,
                'starting_nfelo' : numpy.nan,
                'ending_nfelo' : self.initial_elos[team],
                ## rolling error information ##
                'starting_nfelo_adj' : 0,
                'ending_nfelo_adj' : 0,
                'starting_market_se' : 0,
                'ending_market_se' : 0,
                'starting_model_se' : 0,
                'ending_model_se' : 0,
            }
        ## return ##
        return current_elos
    
    def project_game(self, row):
        '''
        Creates a set of elo projections for a standard "current file" row

        The update process is split so the same logic can be used to project unplayed
        games
        '''
        ## for concision, pull out some local vars from the row ##
        home_team = row['home_team']
        away_team = row['away_team']
        row['starting_nfelo_home'] = self.current_elos[home_team]['ending_nfelo']
        row['starting_nfelo_away'] = self.current_elos[away_team]['ending_nfelo']
        ## offseason regression if necessary ##
        for team_type in ['home', 'away']:
            if row['game_number_{0}'.format(team_type)] == 1 and row['season'] > self.first_season:
                ## some data to keep track of conversions ##
                previous_season_elo = row['starting_nfelo_{0}'.format(team_type)]
                previous_elo_norm = 1505 + (
                    previous_season_elo - statistics.median(self.yearly_elos[row['season']-1])
                )
                ## actual reversion ##
                row['starting_nfelo_{0}'.format(team_type)] = offseason_regression(
                    league_elo = statistics.median(self.yearly_elos[row['season']-1]),
                    previous_elo = row['starting_nfelo_{0}'.format(team_type)],
                    proj_dvoa = row['{0}_projected_dvoa'.format(team_type)],
                    proj_wt_rating = row['{0}_wt_rating'.format(team_type)],
                    reversion = self.config['reversion'],
                    dvoa_weight = self.config['dvoa_weight'],
                    wt_weight = self.config['wt_ratings_weight']
                )
                ## keep track of reversion history ##
                self.reversion_records.append({
                    'team' : row['{0}_team'.format(team_type)],
                    'season' : row['season'],
                    'week' : row['week'],
                    'previous_ending_elo' : previous_season_elo,
                    'league_elo' : statistics.median(self.yearly_elos[row['season']-1]),
                    'mean_reverted_elo' : (
                        self.config['reversion'] * 1505 +
                        (1 - self.config['reversion']) * previous_elo_norm
                    ),
                    'dvoa_elo' : 1505 + 484 * row['{0}_projected_dvoa'.format(team_type)],
                    'wt_elo' : 1505 + 24.8 * row['{0}_wt_rating'.format(team_type)],
                    'new_elo' : row['starting_nfelo_{0}'.format(team_type)]
                })
        ## save an unadjusted elo dif ##
        row['nfelo_dif_pre_adjustment'] = row['starting_nfelo_home'] - row['starting_nfelo_away']
        ## create initial elo dif with game context ##
        initial_elo_dif = (
            ## elo dif #
            row['starting_nfelo_home'] - row['starting_nfelo_away'] +
            ## hfa mod ##
            row['hfa_mod'] +
            ## qb adj ##
            self.config['qb_weight'] * (
                row['home_538_qb_adj'] - row['away_538_qb_adj']
            )
        )
        ## save some output for down stream pipes ##
        row['home_net_qb_mod'] = self.config['qb_weight'] * (
            row['home_538_qb_adj'] - row['away_538_qb_adj']
        )
        row['home_net_bye_mod'] = row['home_bye_mod'] - row['away_bye_mod']
        ## if applicable, gross up for playoffs ##
        if row['is_playoffs']:
            initial_elo_dif = initial_elo_dif * (1+self.config['playoff_boost'])
        ## convert initial elo to probability and spread ##
        ## the data loader already does this for the market open and close
        row['nfelo_home_probability_base'] = elo_to_prob(
            elo_dif=initial_elo_dif,
            z=self.config['z']
        )
        row['nfelo_home_line_base'] = probability_to_spread(
            row['nfelo_home_probability_base']
        )
        row['nfelo_spread_delta'] = row['nfelo_home_line_base'] - row['home_line_open']
        ## calc regressions ##
        mr_regression_open = regress_to_market(
            ## elo difs and lines ##
            initial_elo_dif, row['market_elo_dif_open'],
            row['nfelo_home_line_base'], row['home_line_open'],
            ## regression config ##
            self.config['market_regression'], self.config['min_mr'],
            self.config['spread_delta_base'], self.config['rmse_base'],
            self.config['long_line_inflator'], self.config['hook_certainty'],
            ## errors ##
            self.current_elos[home_team]['ending_model_se'],
            self.current_elos[home_team]['ending_market_se'],
            self.current_elos[away_team]['ending_model_se'],
            self.current_elos[away_team]['ending_market_se'],
        )
        mr_regression_close = regress_to_market(
            ## elo difs and lines ##
            initial_elo_dif, row['market_elo_dif_close'],
            row['nfelo_home_line_base'], row['home_line_close'],
            ## regression config ##
            self.config['market_regression'], self.config['min_mr'],
            self.config['spread_delta_base'], self.config['rmse_base'],
            self.config['long_line_inflator'], self.config['hook_certainty'],
            ## errors ##
            self.current_elos[home_team]['ending_model_se'],
            self.current_elos[home_team]['ending_market_se'],
            self.current_elos[away_team]['ending_model_se'],
            self.current_elos[away_team]['ending_market_se'],
        )
        ## unpack ##
        row['nfelo_dif_base'] = initial_elo_dif
        row['nfelo_dif_open'] = mr_regression_open[0]
        row['market_regression_factor_open'] = mr_regression_open[1]
        row['nfelo_dif_close'] = mr_regression_close[0]
        row['market_regression_factor_close'] = mr_regression_close[1]
        ## translate to spread and win probs ##
        row['nfelo_home_probability_open'] = elo_to_prob(row['nfelo_dif_open'])
        row['nfelo_home_line_open'] = probability_to_spread(row['nfelo_home_probability_open'])
        row['nfelo_home_probability_close'] = elo_to_prob(row['nfelo_dif_close'])
        row['nfelo_home_line_close'] = probability_to_spread(row['nfelo_home_probability_close'])
        ## calc evs ##
        ## Open ## 
        (
            row['home_loss_prob_open'], row['home_push_prob_open'], row['home_cover_prob_open']
        ) = calc_cover_probs(
            row['nfelo_home_line_open'], row['home_line_open']
        )
        row['home_open_ev'] = (row['home_cover_prob_open'] - 1.1 * row['home_loss_prob_open']) / 1.1
        row['away_open_ev'] = (row['home_loss_prob_open'] - 1.1 * row['home_cover_prob_open']) / 1.1
        ## close ##
        (
            row['home_loss_prob_close'], row['home_push_prob_close'], row['home_cover_prob_close']
        ) = calc_cover_probs(
            row['nfelo_home_line_close'], row['home_line_close']
        )
        ## add aways for down stream pipes ##
        row['away_loss_prob_close'] = row['home_cover_prob_close']
        row['away_push_prob_close'] = row['home_push_prob_close']
        row['away_cover_prob_close'] = row['home_loss_prob_close']
        row['home_close_ev'] = (row['home_cover_prob_close'] - 1.1 * row['home_loss_prob_close']) / 1.1
        row['away_close_ev'] = (row['home_loss_prob_close'] - 1.1 * row['home_cover_prob_close']) / 1.1
        ## calc clvs ##
        row['home_clv_from_open'], row['away_clv_from_open'] = calc_clv(
            original_home_spread=row['home_line_open'],
            current_home_spread=row['home_line_close']
        )
        ## save some more down stream datapoints ##
        ## return the row ##
        return row
    
    def process_game(self, row):
        '''
        Process a played row from the current file and updates the elo model
        '''
        ## check for result ##
        if pd.isnull(row['home_margin']) or pd.isnull(row['away_margin']):
            raise Exception('NFELO PROCESS ERROR: Attempted to process an unplayed game')
        ## calculate shifts ##
        ## home ##
        weighted_shift_home = calc_weighted_shift(
            ## observations ##
            [
                (row['home_margin'], self.config['margin_weight']),
                (row['home_net_wepa_point_margin'], self.config['wepa_weight']),
                (row['home_pff_point_margin'], self.config['pff_weight'])
            ],
            ## expectations
            row['nfelo_home_line_base'],
            row['home_line_close'],
            ## 
            self.config['k'], self.config['b'],
            self.config['market_resist_factor'], True
        )
        ## away ##
        weighted_shift_away = calc_weighted_shift(
            ## observations ##
            [
                (row['away_margin'], self.config['margin_weight']),
                (row['away_net_wepa_point_margin'], self.config['wepa_weight']),
                (row['away_pff_point_margin'], self.config['pff_weight'])
            ],
            ## expectations
            row['nfelo_home_line_base'],
            row['home_line_close'],
            ## 
            self.config['k'], self.config['b'],
            self.config['market_resist_factor'], False
        )
        ## apply shift ##
        row['ending_nfelo_home'] = row['starting_nfelo_home'] + weighted_shift_home
        row['ending_nfelo_away'] = row['starting_nfelo_away'] + weighted_shift_away
        ## add errors ##
        row['se_market'] = (row['home_margin'] - row['home_line_close']) ** 2
        row['se_model'] = (row['home_margin'] - row['nfelo_home_line_base']) ** 2
        ## update team records ##
        ## pull out teams and params for concision ##
        ht = row['home_team']
        at = row['away_team']
        adj_alpha = 2 / (1 + self.config['nfelo_span'])
        se_alpha = 2 / (1 + self.config['se_span'])
        ## home ##
        ## basic ##
        self.current_elos[ht]['season'] = row['season']
        self.current_elos[ht]['week'] = row['week']
        self.current_elos[ht]['game_id'] = row['game_id']
        self.current_elos[ht]['opponent'] = row['away_team']
        self.current_elos[ht]['starting_nfelo'] = row['starting_nfelo_home']
        self.current_elos[ht]['ending_nfelo'] = row['ending_nfelo_home']
        ## reset rolling info ##
        self.current_elos[ht]['starting_nfelo_adj'] = self.current_elos[ht]['ending_nfelo_adj']
        self.current_elos[ht]['starting_model_se'] = self.current_elos[ht]['ending_model_se']
        self.current_elos[ht]['starting_market_se'] = self.current_elos[ht]['ending_market_se']
        ## update ##
        self.current_elos[ht]['ending_nfelo_adj'] = (
             self.current_elos[ht]['starting_nfelo_adj'] * (1-adj_alpha) +
             abs(weighted_shift_home) * adj_alpha
        )
        self.current_elos[ht]['ending_model_se'] = (
             self.current_elos[ht]['starting_model_se'] * (1-se_alpha) +
             abs(row['se_model']) * se_alpha
        )
        self.current_elos[ht]['ending_market_se'] = (
             self.current_elos[ht]['starting_market_se'] * (1-se_alpha) +
             abs(row['se_market']) * se_alpha
        )
        ## away ##
        ## basic ##
        self.current_elos[at]['season'] = row['season']
        self.current_elos[at]['week'] = row['week']
        self.current_elos[at]['game_id'] = row['game_id']
        self.current_elos[ht]['opponent'] = row['home_team']
        self.current_elos[at]['starting_nfelo'] = row['starting_nfelo_away']
        self.current_elos[at]['ending_nfelo'] = row['ending_nfelo_away']
        ## reset rolling info ##
        self.current_elos[at]['starting_nfelo_adj'] = self.current_elos[at]['ending_nfelo_adj']
        self.current_elos[at]['starting_model_se'] = self.current_elos[at]['ending_model_se']
        self.current_elos[at]['starting_market_se'] = self.current_elos[at]['ending_market_se']
        ## update ##
        self.current_elos[at]['ending_nfelo_adj'] = (
             self.current_elos[at]['starting_nfelo_adj'] * (1-adj_alpha) +
             abs(weighted_shift_away) * adj_alpha
        )
        self.current_elos[at]['ending_model_se'] = (
             self.current_elos[at]['starting_model_se'] * (1-se_alpha) +
             abs(row['se_model']) * se_alpha
        )
        self.current_elos[at]['ending_market_se'] = (
             self.current_elos[at]['starting_market_se'] * (1-se_alpha) +
             abs(row['se_market']) * se_alpha
        )
        ## save team records ##
        self.elo_records.append(self.current_elos[ht].copy())
        self.elo_records.append(self.current_elos[at].copy())
        ## if it's week 17, append elos to yearly elos for SoS normalization ##
        if row['week'] == 17:
            ## init if needed ##
            if row['season'] not in self.yearly_elos.keys():
                self.yearly_elos[row['season']] = []
            ## update ##
            self.yearly_elos[row['season']].append(row['ending_nfelo_home'])
            self.yearly_elos[row['season']].append(row['ending_nfelo_away'])
        ## return ##
        return row
    
    def apply_nfelo(self, row):
        '''
        Wrapper to project a game, calcualte shifts, and update elo.
        '''
        ## add projections ##
        row = self.project_game(row)
        ## Calculate Shifts ##
        row = self.process_game(row)
        ## return ##
        return row

    def run(self):
        '''
        Primary function for updating the elo model
        '''
        ## filter current down to last completed week ##
        played = self.current_file[
            (
                (self.current_file['week'] <= self.data.last_completed_week) &
                (self.current_file['season'] == self.data.last_completed_season)    
            ) |
            (self.current_file['season'] < self.data.last_completed_season)
        ].copy()
        self.updated_file = played.apply(self.apply_nfelo, axis=1)
    
    def save_reversions(self):
        '''
        Save off season reversions
        '''
        reversions = pd.DataFrame(self.reversion_records)
        reversions.to_csv(
            '{0}/Data/Intermediate Data/offseason_reversions.csv'.format(
                pathlib.Path(__file__).parent.parent.resolve()
            )
        )
    
    def update_config(self, new_config):
        '''
        Re inits the model with a new config, without requiring
        a reloading of data or reinitialization of the instance

        This is used for performance in the optimizer
        '''
        ## update values ##
        for k,v in new_config.items():
            self.config[k] = v
        ## reinit class props to ensure clean dataset ##
        self.current_file = self.data.current_file[
            self.data.current_file['season'] >= self.first_season
        ].copy()
        self.current_elos = self.init_elos()
        self.yearly_elos = {}
        self.reversion_records = []
        self.elo_records = []
        self.updated_file = None

    def project_week(self, unplayed_df):
        '''
        Projects a week of unplayed games based on the current state and status of the model.
        There are currently no checks that the model is updated through the necessary date to make the
        projection
        '''
        ## project games ##
        projected_df = unplayed_df.apply(self.project_game, axis=1)
        ## return ##
        return projected_df
    
    def project_spreads(self):
        '''
        Projects the next unplayed week 
        '''
        ## get unplayed weeks ##
        unplayed = self.data.current_file[
            (self.data.current_file['season'] >= self.first_season) &
            (pd.isnull(self.data.current_file['home_margin']))
        ].groupby(['season', 'week']).head(1)
        ## check that there are games ##
        if len(unplayed) == 0:
            print('Warning -- No unplayed week to project!')
            return
        ## get just the current unplayed ##
        current_unplayed = self.data.current_file[
            (self.data.current_file['week'] == unplayed.iloc[0]['week']) &
            (self.data.current_file['season'] == unplayed.iloc[0]['season'])
        ].copy()
        ## apply nfelo ##
        print('Projecting Week {0}, {1}'.format(
            unplayed.iloc[0]['week'],
            unplayed.iloc[0]['season']
        ))
        self.projections = self.project_week(current_unplayed)

    def extend_updated_file(self):
        '''
        extends the updated file (which only contains played games) to include projections for unplayed games
        '''
        ## get next unplayed week if it exists ##
        ## get unplayed weeks ##
        unplayed = self.data.current_file[
            (self.data.current_file['season'] >= self.first_season) &
            (pd.isnull(self.data.current_file['home_margin']))
        ].groupby(['season', 'week']).head(1)
        ## check that there are games ##
        if len(unplayed) > 0:
            ## get just the current unplayed ##
            current_unplayed = self.data.current_file[
                (self.data.current_file['week'] == unplayed.iloc[0]['week']) &
                (self.data.current_file['season'] == unplayed.iloc[0]['season'])
            ].copy()
            ## project ##
            projected = self.project_week(current_unplayed)
            ## merge ##
            self.updated_file_ext = pd.concat([self.updated_file, projected])
        else:
            self.updated_file_ext = self.updated_file.copy()
        ## reset index that may get stacked with the merge ##
        self.updated_file_ext.reset_index(drop=True, inplace=True)
        

