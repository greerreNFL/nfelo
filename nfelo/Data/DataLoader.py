import pandas as pd
import numpy
import pathlib

import nfelodcm as dcm

from ..Utilities import (
    american_to_hold_adj_prob, spread_to_probability,
    prob_to_elo, merge_check, probability_to_spread
)

class DataLoader:
    '''
    Loads and formats all data for the model update using nfelodcm
    '''
    def __init__(self):
        self.package_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
        self.intermediate_data_loc = '{0}/Intermediate Data'.format(pathlib.Path(__file__).parent.resolve())
        print('Loading data...')
        self.last_completed_season, self.last_completed_week = dcm.get_season_state()
        self.db = dcm.load([
            'games', 'rosters', 'logos', ## fastr ##
            'wepa', 'wt_ratings', 'hfa', 'qbelo', ## nfelo models
            'filmmargins', 'market_data' ## other data
        ])
        self.dvoa_projections = pd.read_csv(
            '{0}/dvoa_projections.csv'.format(self.intermediate_data_loc),
            index_col=0
        )
        self.market_data = self.format_market_data()
        self.current_file = self.gen_current_file()

    def format_market_data(self):
        '''
        Formats market_data and adds probability columns
        '''
        print('Formatting market data...')
        ## copy & rename ##
        market_data = self.db['market_data'][[
            ## meta ##
            'game_id', 'season', 'week', 'home_team', 'away_team',
            ## spreads ##
            'home_spread_open', 'home_spread_open_price', 'away_spread_open_price',
            'home_spread_last', 'home_spread_last_price', 'away_spread_last_price',
            'home_spread_tickets_pct',
            ## ml ##
            'home_ml_open', 'away_ml_open',
            'home_ml_last', 'away_ml_last',
            ## total ##
            'total_line_open', 'under_price_open', 'over_price_open',
            'total_line_last', 'under_price_last', 'over_price_last',
        ]].copy().rename(columns={
            'home_spread_open' : 'home_line_open',
            'home_spread_open_price' : 'home_line_open_price',
            'away_spread_open_price' : 'away_line_open_price',
            'home_spread_last' : 'home_line_close',
            'home_spread_last_price' : 'home_line_close_price',
            'away_spread_last_price' : 'away_line_close_price',
            'home_spread_tickets_pct' : 'home_ats_pct',
            'home_ml_last' : 'home_ml_close',
            'away_ml_last' : 'away_ml_close',
            'total_line_last' : 'total_line_close',
            'under_price_last' : 'under_price_close',
            'over_price_last' : 'over_price_close',
        })
        ## fill missing closes ##
        if len(market_data[pd.isnull(market_data['home_line_close'])]) > 0:
            print('     Warning -- {0} games were missing a spread'.format(
                len(market_data[pd.isnull(market_data['home_line_close'])])
            ))
            print('                /n'.join(
                market_data[pd.isnull(market_data['home_line_close'])]['game_id'].tolist()
            ))
            print('                Filling with 0...')
        market_data['home_line_close'] = market_data['home_line_close'].fillna(0)
        ## fill missing opens ##
        market_data['home_line_open'] = market_data['home_line_open'].fillna(
            market_data['home_line_close']
        )
        ## add ml wps ##
        (
            market_data['ml_implied_home_win_probability_open'],
            market_data['ml_implied_away_win_probability_open'],
            market_data['ml_hold_open']
        ) = american_to_hold_adj_prob(market_data['home_ml_open'], market_data['away_ml_open'])
        (
            market_data['ml_implied_home_win_probability_close'],
            market_data['ml_implied_away_win_probability_close'],
            market_data['ml_hold_close']
        ) = american_to_hold_adj_prob(market_data['home_ml_close'], market_data['away_ml_close'])
        ## add spread implied wps ##
        market_data['spread_implied_home_win_probability_open'] = spread_to_probability(
            market_data['home_line_open']
        )
        market_data['spread_implied_away_win_probability_open'] = 1 - market_data['spread_implied_home_win_probability_open']
        market_data['spread_implied_home_win_probability_close'] = spread_to_probability(
            market_data['home_line_close']
        )
        market_data['spread_implied_away_win_probability_close'] = 1 - market_data['spread_implied_home_win_probability_close']
        ## combine implied
        market_data['home_implied_win_probability_open'] = market_data['spread_implied_home_win_probability_open'].combine_first(
            market_data['ml_implied_home_win_probability_open']
        )
        market_data['away_implied_win_probability_open'] = 1- market_data['home_implied_win_probability_open']
        market_data['home_implied_win_probability_close'] = market_data['spread_implied_home_win_probability_close'].combine_first(
            market_data['ml_implied_home_win_probability_close']
        )
        market_data['away_implied_win_probability_close'] = 1- market_data['home_implied_win_probability_close']
        ## add implied elo dif ##
        market_data['home_implied_elo_dif_open'] = prob_to_elo(market_data['home_implied_win_probability_open'])
        market_data['home_implied_elo_dif_close'] = prob_to_elo(market_data['home_implied_win_probability_close'])
        market_data.to_csv(
            '{0}/market_data.csv'.format(self.intermediate_data_loc)
        )
        ## return ##
        return market_data

    def format_games(self):
        '''
        Format the game file
        '''
        print('Merging game data with dvoa, pff, market data, etc...')
        ## needed cols ##
        games = self.db['games'][[
            'game_id','game_type','season','week',
            'home_team','away_team','home_score','away_score',
            'gameday', 'weekday' ,'stadium','stadium_id', 'old_game_id',
        ]].rename(columns={
            'gameday' : 'game_date',
            'weekday' : 'game_day'
        }).copy()
        ## rename game_type to match traditional styling
        games['game_type'] = games['game_type'].replace({
            'REG' : 'reg',
            'WC' : 'post',
            'DIV' : 'post',
            'CON' : 'post',
            'SB' : 'post',
        })
        games = games.rename(columns={'game_type' : 'type'})
        ## add some aditional fields needed downstream ##
        games['home_margin'] = games['home_score'] - games['away_score']
        games['away_margin'] = games['away_score'] - games['home_score']
        games['is_playoffs'] = numpy.where(
            games['type'] == 'post',
            1,
            0
        )
        ## return ##
        return games

    def add_market_info(self, games):
        '''
        Adds market data to the games file
        '''
        ## add market info ##
        games = pd.merge(
            games,
            self.market_data[[
                'game_id', 'home_line_open', 'home_line_close',
                'home_line_close_price', 'away_line_close_price',
                'home_ml_close', 'away_ml_close',
                'home_ats_pct', 'home_implied_win_probability_open',
                'home_implied_win_probability_close', 'home_implied_elo_dif_open',
                'home_implied_elo_dif_close'
            ]].rename(columns={
                'home_ml_close' : 'home_moneyline',
                'away_ml_close' : 'away_moneyline',
                'home_line_close_price' : 'home_spread_odds',
                'away_line_close_price' : 'away_spread_odds',
                'home_ats_pct' : 'ats_pct',
                'home_implied_win_probability_open' : 'market_home_probability_open',
                'home_implied_win_probability_close' : 'market_home_probability_close',
                'home_implied_elo_dif_open' : 'market_elo_dif_open',
                'home_implied_elo_dif_close' : 'market_elo_dif_close',
            }),
            on=['game_id'],
            how='left'
        )
        ## return ##
        return games
    
    def add_wt_ratings(self, games):
        '''
        Adds win total ratings for pre-seaosn and in season
        '''
        games = pd.merge(
            games,
            self.db['wt_ratings'][[
                'team', 'season', 'wt_rating', 'wt_rating_elo'
            ]].rename(columns={
                'team' : 'home_team',
                'wt_rating' : 'home_wt_rating',
                'wt_rating_elo' : 'home_wt_rating_elo'
            }),
            on=['home_team', 'season'],
            how='left'
        )
        games = pd.merge(
            games,
            self.db['wt_ratings'][[
                'team', 'season', 'wt_rating', 'wt_rating_elo'
            ]].rename(columns={
                'team' : 'away_team',
                'wt_rating' : 'away_wt_rating',
                'wt_rating_elo' : 'away_wt_rating_elo'
            }),
            on=['away_team', 'season'],
            how='left'
        )
        ## return ##
        return games
    
    def add_dvoa(self, games):
        '''
        Adds projected dvoa to the games file
        '''
        ## combine DVOA ##
        games = pd.merge(
            games,
            self.dvoa_projections[[
                'team', 'season', 'projected_total_dvoa'
            ]].rename(columns={
                'team' : 'home_team',
                'projected_total_dvoa' : 'home_projected_dvoa',
            }),
            on=['home_team', 'season'],
            how='left'
        )
        games = pd.merge(
            games,
            self.dvoa_projections[[
                'team', 'season', 'projected_total_dvoa'
            ]].rename(columns={
                'team' : 'away_team',
                'projected_total_dvoa' : 'away_projected_dvoa',
            }),
            on=['away_team', 'season'],
            how='left'
        )
        ## return ##
        return games
    
    def add_wepa_margins(self, games):
        '''
        Adds the wepa margins to the game file
        '''
        ## home team ##
        games = pd.merge(
            games,
            self.db['wepa'][[
                'game_id', 'team', 'wepa', 'd_wepa', 'wepa_net',
                'epa', 'epa_against', 'epa_net'
            ]].rename(columns={
                'team' : 'home_team',
                'wepa' : 'home_offensive_wepa',
                'd_wepa' : 'home_defensive_wepa',
                'wepa_net' : 'home_net_wepa',
                'epa' : 'home_offensive_epa',
                'epa_against' : 'home_defensive_epa',
                'epa_net' : 'home_net_epa'
            }),
            on=['game_id', 'home_team'],
            how='left'
        )
        ## home team ##
        games = pd.merge(
            games,
            self.db['wepa'][[
                'game_id', 'team', 'wepa', 'd_wepa', 'wepa_net',
                'epa', 'epa_against', 'epa_net'
            ]].rename(columns={
                'team' : 'away_team',
                'wepa' : 'away_offensive_wepa',
                'd_wepa' : 'away_defensive_wepa',
                'wepa_net' : 'away_net_wepa',
                'epa' : 'away_offensive_epa',
                'epa_against' : 'away_defensive_epa',
                'epa_net' : 'away_net_epa'
            }),
            on=['game_id', 'away_team'],
            how='left'
        )
        ## create another field for wepa margin 
        ## for now, implimented as a copy of wepa net, but
        ## in the future will be a regressed normalization
        games['home_net_wepa_point_margin'] = games['home_net_wepa']
        games['away_net_wepa_point_margin'] = games['away_net_wepa']
        ## return ##
        return games
    
    def add_pff_margins(self, games):
        '''
        Adds pff margins to the game file
        '''
        ## combine PFF
        games = pd.merge(
            games,
            self.db['filmmargins'][[
                'game_id', 'team', 'film_margin'
            ]].rename(columns={
                'team' : 'home_team',
                'film_margin' : 'home_pff_point_margin',
            }),
            on=['game_id', 'home_team'],
            how='left'
        )
        games = pd.merge(
            games,
            self.db['filmmargins'][[
                'game_id', 'team', 'film_margin'
            ]].rename(columns={
                'team' : 'away_team',
                'film_margin' : 'away_pff_point_margin',
            }),
            on=['game_id', 'away_team'],
            how='left'
        )
        ## fill missing ##
        hm = games['home_score'] - games['away_score']
        am = games['away_score'] - games['home_score']
        games['home_pff_point_margin'] = games['home_pff_point_margin'].fillna(hm)
        games['away_pff_point_margin'] = games['away_pff_point_margin'].fillna(am)
        ## return ##
        return games
    
    def add_hfa(self, games):
        '''
        Adds hfa data to the game file
        '''
        games = pd.merge(
            games,
            ## need to add neutral fields to hfa model ##
            self.db['hfa'][[
                'game_id', 'home_bye', 'away_bye', 'gametime', 'location', 'roof',
                'surface', 'temp', 'wind', 'home_time_advantage',
                'dif_surface', 'div_game', 'hfa_base', 'home_bye_adj',
                'away_bye_adj', 'home_time_advantage_adj', 'dif_surface_adj',
                'div_game_adj', 'hfa_adj'
            ]].rename(columns={
                'temp' : 'temperature'
            }),
            on=['game_id'],
            how='left'
        )
        games['is_neutral'] = numpy.where(
            games['location'] == 'Neutral',
            1,
            0
        )
        ## translate to elo ##
        games['hfa_base_mod'] = 25 * games['hfa_base']
        for mod in [
            'home_bye', 'away_bye', 'home_time_advantage',
            'dif_surface', 'div_game', 'hfa'
        ]:
            games['{0}_mod'.format(mod)] = 25 * games['{0}_adj'.format(mod)] 
        ## return ##
        return games

    def add_qbs(self, games):
        '''
        Add 538 data to games
        '''
        games = pd.merge(
            games,
            self.db['qbelo'][[
                'game_id', 'elo1_pre', 'elo1_post',
                'elo2_pre', 'elo2_post', 'qbelo1_pre',
                'qbelo1_post', 'qbelo2_pre', 'qbelo2_post',
                'qb1', 'qb2', 'qb1_adj', 'qb2_adj',
                'elo_prob1', 'qbelo_prob1'
            ]].rename(columns={
                'elo1_pre' : 'home_elo_pre',
                'elo1_post' : 'home_elo_post',
                'elo2_pre' : 'away_elo_pre',
                'elo2_post' : 'away_elo_post',
                'qbelo1_pre' : 'home_qbelo_pre',
                'qbelo1_post' : 'home_qbelo_post',
                'qbelo2_pre' : 'away_qbelo_pre',
                'qbelo2_post' : 'away_qbelo_post',
                'qb1' : 'home_538_qb',
                'qb2' : 'away_538_qb',
                'qb1_adj' : 'home_538_qb_adj',
                'qb2_adj' : 'away_538_qb_adj'
            }),
            on=['game_id'],
            how='left'
        )
        ## fill any missing ##
        missing = games[
            (pd.isnull(games['home_538_qb_adj'])) |
            (pd.isnull(games['away_538_qb_adj']))
        ].copy()
        ## add spreads ##
        games['538_home_line_close'] = probability_to_spread(games['elo_prob1'])
        games['qbelo_home_line_close'] = probability_to_spread(games['qbelo_prob1'])
        if len(missing) > 0:
            print('          Warning - {0} games were missing qb adjs'.format(
                len(missing)
            ))
            print('                    Filling with 0')
            games['home_538_qb_adj'] = games['home_538_qb_adj'].fillna(0)
            games['away_538_qb_adj'] = games['away_538_qb_adj'].fillna(0)
        ## return ##
        return games

    def add_game_numbers(self, games):
        '''
        Adds game numbers to the games file
        '''
        ## flatten the games file to get a unique set of team games ##
        flat = pd.concat([
            self.db['games'][['home_team', 'season', 'week', 'game_id']].copy().rename(columns={
                'home_team' : 'team'
            }),
            self.db['games'][['away_team', 'season', 'week', 'game_id']].copy().rename(columns={
                'away_team' : 'team'
            })
        ]).sort_values(
            by=['team', 'season', 'week'],
            ascending=[True, True, True]
        ).reset_index(drop=True)
        ## add game numbers to flat ##
        flat['game_number_all_time'] = flat.groupby(['team']).cumcount() + 1
        flat['game_number'] = flat.groupby(['team', 'season']).cumcount() + 1
        ## add prev and next game numbers ##
        flat['game_id_previous'] = flat.groupby(['team'])['game_id'].shift(1)
        flat['game_id_next'] = flat.groupby(['team'])['game_id'].shift(-1)
        flat['week_previous'] = flat.groupby(['team'])['week'].shift(1)
        flat['week_next'] = flat.groupby(['team'])['week'].shift(-1)
        ## add back to games ##
        games = pd.merge(
            games,
            flat.rename(columns={
                'team' : 'home_team',
                'game_number_all_time' : 'all_time_game_number_home',
                'game_number' : 'game_number_home',
                'game_id_previous' : 'prev_game_id_home',
                'game_id_next' : 'next_game_id_home',
                'week_previous' : 'prev_week_home',
                'week_next' : 'next_week_home'
            }),
            on=['home_team', 'season', 'week', 'game_id'],
            how='left'
        )
        games = pd.merge(
            games,
            flat.rename(columns={
                'team' : 'away_team',
                'game_number_all_time' : 'all_time_game_number_away',
                'game_number' : 'game_number_away',
                'game_id_previous' : 'prev_game_id_away',
                'game_id_next' : 'next_game_id_away',
                'week_previous' : 'prev_week_away',
                'week_next' : 'next_week_away'
            }),
            on=['away_team', 'season', 'week', 'game_id'],
            how='left'
        )
        ## return ##
        return games

    def gen_current_file(self):
        '''
        Wrapper that generates the current file by first formating the games file and then
        merging market data, dvoa, etc etc
        '''
        ## get games ##
        games = self.format_games()
        ## perform merges ##
        games = merge_check(self.add_game_numbers, games, 'game numbers')
        games = merge_check(self.add_market_info, games, 'market data')
        games = merge_check(self.add_wt_ratings, games, 'wt ratings')
        games = merge_check(self.add_dvoa, games, 'dvoa projections')
        games = merge_check(self.add_wepa_margins, games, 'wepa margins')
        games = merge_check(self.add_pff_margins, games, 'pff margins')
        games = merge_check(self.add_hfa, games, 'hfa')
        games = merge_check(self.add_qbs, games, 'qbs')
        ## save ##
        games.to_csv(
            '{0}/current_file.csv'.format(self.intermediate_data_loc)
        )
        ## return ##
        return games


