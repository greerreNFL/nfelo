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

file_loc = config['formatting']['spreads']['fastr_line_loc']
alt_loc = config['formatting']['spreads']['alt_line_loc']
tfl_loc = config['formatting']['spreads']['tfl_line_loc']
vi_loc = config['formatting']['spreads']['vi_line_loc']
mult_loc = config['formatting']['spreads']['mult_loc']
teams = config['formatting']['spreads']['teams']
headers = config['formatting']['spreads']['headers']

## create output sub path ##
output_folder = '/data_sources/formatted'


def format_spreads():
    print('Merging and formatting market data...')
    print('     Loading fastr lines...')
    ## load game data ##
    game_df = pd.read_csv(
        '{0}/{1}'.format(
            package_dir,
            file_loc
        ),
        index_col=0
    )
    ## filter out probowls ##
    game_df = game_df[
        numpy.isin(
            game_df['home_team'],
            teams
        )
    ]
    ## format based on how future scripts will read ##
    game_df['home_spread_fastR'] = game_df['spread_line'] * -1
    game_df['home_moneyline_fastR'] = game_df['home_moneyline']
    game_df['away_moneyline_fastR'] = game_df['away_moneyline']
    game_df['total_fastR'] = game_df['total_line']
    game_df['neutral_field'] = numpy.where(game_df['location'] == 'Neutral', 1,0)
    game_df = game_df.rename(columns={'div_game' : 'divisional_game'})
    ## load alt_lines ##
    print('     Loading SBR lines...')
    alt_df = pd.read_csv(
        '{0}/{1}'.format(
            package_dir,
            alt_loc
        ),
        index_col=0
    )
    ## group to make sure we aren't adding any dupes ##
    alt_df = alt_df.groupby(['game_id']).head(1)
    print('     Loading TFL lines...')
    tfl_df = pd.read_csv(
        '{0}/{1}'.format(
            package_dir,
            tfl_loc
        ),
        index_col=0
    )
    tfl_df['type'] = numpy.where(
        tfl_df['season'] > 2020,
        numpy.where(
            tfl_df['week'] == 19,
            'post',
            'reg'
        ),
        numpy.where(
            tfl_df['week'] == 18,
            'post',
            'reg'
        )
    )
    ## remove games w/ bad data ##
    tfl_df = tfl_df[~((tfl_df['season'] == 2017) & (tfl_df['week'] == 1) & (tfl_df['home_team'] == 'MIA'))]
    tfl_df = tfl_df[~((tfl_df['season'] == 2020) & (tfl_df['week'] == 1) & (tfl_df['home_team'] == 'SF'))]
    tfl_df = tfl_df[~((tfl_df['season'] == 2019) & (tfl_df['week'] == 1) & (tfl_df['home_team'] == 'OAK'))]
    ## group to make sure we aren't adding any dupes ##
    tfl_df = tfl_df.groupby(
        ['season', 'type', 'home_team', 'away_team']
    ).head(1)
    print('     Loading VI lines...')
    vi_df = pd.read_csv(
        '{0}/{1}'.format(
            package_dir,
            vi_loc
        ),
        index_col=0
    )
    ## group to make sure we aren't adding any dupes ##
    vi_df = vi_df.groupby(
        ['season', 'week', 'home_team', 'away_team']
    ).head(1)
    ## join ##
    print('     Merging data...')
    game_df = pd.merge(
        game_df,
        alt_df[[
            'game_id',
            'sbr_home_closing_line',
            'sbr_home_closing_price',
            'sbr_away_closing_line',
            'sbr_away_closing_price',
            'sbr_home_opening_line',
            'sbr_home_opening_price',
            'sbr_away_opening_line',
            'sbr_away_opening_price',
        ]].rename(columns={
            'sbr_home_closing_line' : 'closing_home_line_sbr',
            'sbr_home_closing_price' : 'closing_home_price_sbr',
            'sbr_away_closing_line' : 'closing_away_line_sbr',
            'sbr_away_closing_price' : 'closing_away_price_sbr',
            'sbr_home_opening_line' : 'opening_home_line_sbr',
            'sbr_home_opening_price' : 'opening_home_price_sbr',
            'sbr_away_opening_line' : 'opening_away_line_sbr',
            'sbr_away_opening_price' : 'opening_away_price_sbr',
        }),
        on=['game_id'],
        how='left'
    )
    game_df = pd.merge(
        game_df,
        tfl_df[[
            'season',
            'home_team',
            'away_team',
            'type',
            'opening_home_line_tfl',
            'closing_home_line_tfl'
        ]],
        on=['season', 'type', 'home_team', 'away_team'],
        how='left'
    )
    game_df = pd.merge(
        game_df,
        vi_df,
        on=['season', 'week', 'home_team', 'away_team'],
        how='left'
    )
    ## merge lines ##
    ## again, we're just using fastr here ##
    print('     Formatting data...')
    game_df['home_line_open'] = game_df['opening_home_line_tfl'].combine_first(
        game_df['opening_home_line_vi']
    ).combine_first(
        game_df['opening_home_line_sbr']
    ).combine_first(
        game_df['home_spread_fastR']
    )
    game_df['home_line_close'] = game_df['closing_home_line_vi'].combine_first(
        game_df['closing_home_line_sbr']
    ).combine_first(
        game_df['home_spread_fastR']
    ).combine_first(
        game_df['closing_home_line_tfl']
    )
    game_df['game_total'] = game_df['total_fastR']
    game_df['home_ats_pct'] = game_df['home_spread_weight_vi']
    ## calc market implied home win prob ##
    print('     Calculating implied win probablities...')
    ## load spead mult dict ##
    spread_multiples_df = pd.read_csv(
        '{0}/{1}'.format(
            package_dir,
            mult_loc
        )
    )
    ## calc ML implied win prob ##
    game_df['home_ml_implied_wp_ex_hold'] = numpy.where(
        game_df['home_moneyline_fastR'] < 0,
        (
            (-1 * game_df['home_moneyline_fastR']) /
            (100 - game_df['home_moneyline_fastR'])
        ),
        (
            100 /
            (100 + game_df['home_moneyline_fastR'])
        )
    )
    game_df['away_ml_implied_wp_ex_hold'] = numpy.where(
        game_df['away_moneyline_fastR'] < 0,
        (
            (-1 * game_df['away_moneyline_fastR']) /
            (100 - game_df['away_moneyline_fastR'])
        ),
        (
            100 /
            (100 + game_df['away_moneyline_fastR'])
        )
    )
    game_df['moneyline_hold'] = (
        1 -
        (game_df['away_ml_implied_wp_ex_hold'] + game_df['home_ml_implied_wp_ex_hold'])
    )
    game_df['home_ml_implied_wp'] = (
        .5 * game_df['moneyline_hold'] +
        game_df['home_ml_implied_wp_ex_hold']
    )
    game_df['away_ml_implied_wp'] = (
        .5 * game_df['moneyline_hold'] +
        game_df['away_ml_implied_wp_ex_hold']
    )
    ## calc spread implied wp ##
    ## create a df with all possible spreads ##
    cur_spread = spread_multiples_df['implied_spread'].min()
    all_spreads = []
    while cur_spread <= spread_multiples_df['implied_spread'].max():
        all_spreads.append({
            'home_line_close' : cur_spread,
        })
        cur_spread += .5
    all_spreads_df = pd.DataFrame(all_spreads)
    ## attach win probs ##
    all_spreads_df = pd.merge(
        all_spreads_df,
        spread_multiples_df[[
            'implied_spread', 'win_prob'
        ]].rename(columns={
            'implied_spread' : 'home_line_close',
        }),
        on=['home_line_close'],
        how='left'
    )
    all_spreads_df['win_prob'] = all_spreads_df['win_prob'].ffill().bfill()
    ## agg ##
    all_spreads_df = all_spreads_df.groupby(['home_line_close']).agg(
        spread_wp_low = ('win_prob', 'min'),
        spread_wp_mid = ('win_prob', 'mean'),
        spread_wp_high = ('win_prob', 'max'),
    ).reset_index()
    ## pull in to game df ##
    game_df = pd.merge(
        game_df,
        all_spreads_df,
        on=['home_line_close'],
        how='left'
    )
    ## get spread implied wp based on juice ##
    game_df['home_spread_implied_wp'] = numpy.where(
        pd.isnull(game_df['home_spread_odds']),
        game_df['spread_wp_mid'],
        numpy.where(
            game_df['home_spread_odds'] > -110,
            game_df['spread_wp_low'],
            numpy.where(
                game_df['home_spread_odds'] == -110,
                game_df['spread_wp_mid'],
                game_df['spread_wp_high']
            )
        )
    )
    ## merge with pref for moneyline ##
    game_df['market_implied_win_probability'] = game_df['home_ml_implied_wp'].combine_first(
        game_df['home_spread_implied_wp']
    )
    ## translate to elo ##
    game_df['market_implied_elo_dif'] = numpy.log10(
        (1 / game_df['market_implied_win_probability']) -
        1
    ) * -400
    ## filter cols ##
    game_df = game_df[headers]
    ## catch for any bad open / closing lines ##
    game_df['home_line_close'] = numpy.maximum(
        -51, numpy.minimum(
            51,
            game_df['home_line_close']
        )
    )
    game_df['home_line_open'] = numpy.where(
        (game_df['home_line_open'] > 51) |
        (game_df['home_line_open'] < -51),
        game_df['home_line_close'],
        game_df['home_line_open']
    )
    ## save ##
    print('     Done!')
    game_df.to_csv(
        '{0}{1}/market_data.csv'.format(
            package_dir,
            output_folder
        )
    )
