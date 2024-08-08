import pandas as pd
import numpy
import json
from scipy.optimize import minimize


package_dir = '/opt/homebrew/lib/python3.11/site-packages/nfelo'
config = None
with open('{0}/config.json'.format(package_dir), 'r') as fp:
    config = json.load(fp)

total_loc = config['models']['wt_ratings']['total_loc']
games_loc = config['models']['wt_ratings']['game_loc']
logos_loc = config['models']['wt_ratings']['logo_loc']
wt_config = config['models']['wt_ratings']['wt_config']
repl = config['data_pulls']['nflfastR']['team_standardization']

output_folder = '/data_sources/formatted'

## helpers ##
def american_to_prob(series):
    ## convert odds to prob ##
    return numpy.where(
        series < 100,
        (-1 * series) / (100 - series),
        100 / (100 + series)
    )

def calc_vf_over_prob(over_prob, under_prob):
    ## calculate a vig
    return (
        over_prob /
        (over_prob + under_prob)
    )

def spread_to_prob(spread):
    ## convert to elo dif ##
    dif = spread * 25
    ## convert to prob ##
    prob = (
        1 /
        (
            10 ** (-dif / 400) +
            1
        )
    )
    return prob

def calc_probs_and_hold(over, under):
    ## takes over and under as american and returns ##
    ## vig free prob and hold ##
    over_prob = american_to_prob(over)
    under_prob = american_to_prob(under)
    vf_over_prob = calc_vf_over_prob(over_prob, under_prob)
    vf_under_prob = 1 - vf_over_prob
    hold = (over_prob + under_prob) - 1
    return vf_over_prob, vf_under_prob, hold


## main funcs ##
def load_dfs():
    ## read in data frames and do any necessary formating ##
    totals = pd.read_csv(
        '{0}/{1}'.format(
            package_dir,
            total_loc
        ),
        index_col=0
    )
    totals['team'] = totals['team'].replace(repl)
    games = pd.read_csv(
        '{0}/{1}'.format(
            package_dir,
            games_loc
        ),
        index_col=0
    )
    logos = pd.read_csv(
        '{0}/{1}'.format(
            package_dir,
            logos_loc
        ),
        index_col=0
    )
    logos['team_abbr'] = logos['team_abbr'].replace(repl)
    ## for results, check existing ##
    existing = None
    try:
        existing = pd.read_csv(
            '{0}/{1}/wt_ratings.csv'.format(
                package_dir,
                output_folder
            ),
            index_col=0
        )
        if existing['season'].max() < games['season'].max():
            print('          Found existing rating through {0}. Will update...'.format(
                existing['season'].max()
            ))
        else:
            print('          Existing WT Ratings are up to date. No update required...')

    except:
        print('          Existing WT ratings not found, will optimize from 2003...')
    ## return ##
    return totals, games, logos, existing


def calc_vig_free_odds(df, wt_config):
    ## calc vig free odds ##
    df['over_prob'] = american_to_prob(df['over_odds'])
    df['under_prob'] = american_to_prob(df['under_odds'])
    df['over_prob_vf'] = (
        df['over_prob'] /
        (df['over_prob'] + df['under_prob'])
    )
    df['under_prob_vf'] = 1 - df['over_prob_vf']
    ## take the logit of the over prob for regression ##
    df['logit_over_prob_vf'] = numpy.log(
        df['over_prob_vf'] /
        (1 - df['over_prob_vf'])
    )
    ## adjust total line by up to a half win based on
    df['line_adj'] = (
        df['line'] +
        df['logit_over_prob_vf'] * wt_config['over_prob_logit_coef']
    )
    ## drop ##
    df = df.drop(columns=['over_prob', 'under_prob'])
    return df


## Opti structure for spread based ratings ##
def construct_vars(season_df):
    ## constructs variables and bounds for optimization ##
    ## first variable is HFA, then rest are team sorted alpha $$
    best_guesses = [
        1.5
    ]
    ## an array to keep track of variable index ##
    var_keys = [
        'HFA'
    ]
    ## list of bounds for optimization ##
    bounds_list = [
        (-5, 5) ## hfa ##
    ]
    ## iter through frame to populate ##
    season_df = season_df.sort_values(
        by=['team'],
        ascending=[True]
    ).reset_index(drop=True)
    for index, row in season_df.iterrows():
        best_guesses.append(row['line_adj'] - 8)
        var_keys.append(row['team'])
        bounds_list.append((-15,15))
    ## return opti config ##
    return best_guesses, var_keys, bounds_list


## applies ratings within the optimization loop ##
def apply_wins(x, var_keys, df):
    ## estimates margin based on team ratings, then calculates
    ## win probability from margin, which is summed to create season
    ## win total estimation ##
    temp = df.copy()
    ## first construct dictionary of values to do vectorized apply ##
    val_dict = {}
    for index, value in enumerate(x):
        val_dict[
            var_keys[index]
        ] = value
    ## then apply to frame with a replace ##
    temp['hfa'] = x[0]
    temp['home_rating'] = temp['home_team'].map(val_dict)
    temp['away_rating'] = temp['away_team'].map(val_dict)
    ## calc margin ##
    temp['expected_margin'] = (
        temp['home_rating'] +
        temp['hfa'] -
        temp['away_rating']
    )
    ## calc win probs from margin ##
    temp['home_win'] = spread_to_prob(temp['expected_margin'])
    temp['away_win'] = 1 - temp['home_win']
    ## retrun ##
    return temp


def score_opti(applied_df, lines):
    ## score optimization with rmse ##
    ## flatten ##
    temp = pd.concat([
        ## home ##
        applied_df[[
            'home_team', 'home_win'
        ]].rename(columns={
            'home_team' : 'team',
            'home_win' : 'win'
        }),
        ## away ##
        applied_df[[
            'away_team', 'away_win'
        ]].rename(columns={
            'away_team' : 'team',
            'away_win' : 'win'
        })
    ])
    ## aggregate ##
    agg = temp.groupby(['team']).agg(
        wins = ('win', 'sum'),
    ).reset_index()
    ## add scores ##
    agg = pd.merge(
        agg,
        lines,
        on=['team'],
        how='left'
    )
    ## calc error ##
    agg['se'] = (
        agg['wins'] -
        agg['line_adj']
    ) ** 2
    rmse = agg['se'].mean() ** (1/2)
    ## return ##
    return rmse


def obj_func(x, lines, games, var_keys):
    ## actual function that optimizes ##
    ## apply wins ##
    applied = apply_wins(x, var_keys, games)
    ## score ##
    rmse = score_opti(
        applied, lines
    )
    ## return rmse as obj func ##
    return rmse



## Opti structure for spread based ratings ##
def construct_vars_elo(season_df):
    ## constructs variables and bounds for optimization ##
    ## first variable is HFA, then rest are team sorted alpha $$
    best_guesses = [
        37.5
    ]
    ## an array to keep track of variable index ##
    var_keys = [
        'HFA'
    ]
    ## list of bounds for optimization ##
    bounds_list = [
        (0, 70) ## hfa ##
    ]
    ## iter through frame to populate ##
    season_df = season_df.sort_values(
        by=['team'],
        ascending=[True]
    ).reset_index(drop=True)
    for index, row in season_df.iterrows():
        best_guesses.append((row['line_adj'] - 8) / 7 * 200 + 1505)
        var_keys.append(row['team'])
        bounds_list.append((-1100,1800))
    ## return opti config ##
    return best_guesses, var_keys, bounds_list


## applies ratings within the optimization loop ##
def apply_wins_elo(x, var_keys, df):
    ## estimates margin based on team ratings, then calculates
    ## win probability from margin, which is summed to create season
    ## win total estimation ##
    temp = df.copy()
    ## first construct dictionary of values to do vectorized apply ##
    val_dict = {}
    for index, value in enumerate(x):
        val_dict[
            var_keys[index]
        ] = value
    ## then apply to frame with a replace ##
    temp['hfa'] = x[0]
    temp['home_rating'] = temp['home_team'].map(val_dict)
    temp['away_rating'] = temp['away_team'].map(val_dict)
    ## calc margin ##
    temp['expected_elo_dif'] = (
        temp['home_rating'] +
        temp['hfa'] -
        temp['away_rating']
    )
    ## calc win probs from margin ##
    temp['home_win'] = 1.0 / (numpy.power(10.0, (-1 * temp['expected_elo_dif']/400)) + 1.0)
    temp['away_win'] = 1 - temp['home_win']
    ## retrun ##
    return temp


def score_opti_elo(applied_df, lines):
    ## score optimization with rmse ##
    ## flatten ##
    temp = pd.concat([
        ## home ##
        applied_df[[
            'home_team', 'home_win'
        ]].rename(columns={
            'home_team' : 'team',
            'home_win' : 'win'
        }),
        ## away ##
        applied_df[[
            'away_team', 'away_win'
        ]].rename(columns={
            'away_team' : 'team',
            'away_win' : 'win'
        })
    ])
    ## aggregate ##
    agg = temp.groupby(['team']).agg(
        wins = ('win', 'sum'),
    ).reset_index()
    ## add scores ##
    agg = pd.merge(
        agg,
        lines,
        on=['team'],
        how='left'
    )
    ## calc error ##
    agg['se'] = (
        agg['wins'] -
        agg['line_adj']
    ) ** 2
    rmse = agg['se'].mean() ** (1/2)
    ## return ##
    return rmse


def obj_func_elo(x, lines, games, var_keys):
    ## actual function that optimizes ##
    ## apply wins ##
    applied = apply_wins_elo(x, var_keys, games)
    ## score ##
    rmse = score_opti_elo(
        applied, lines
    )
    ## return rmse as obj func ##
    return rmse




def calc_sos(x, var_keys, df):
    ## uses wt_ratings and schedule to calculte sos ##
    ## apply ratings to schedule ##
    temp = apply_wins(x, var_keys, df)
    ## flatten ##
    flat = pd.concat([
        temp[[
            'home_team', 'away_rating'
        ]].rename(columns={
            'home_team' : 'team',
            'away_rating' : 'opponent_rating'
        }),
        temp[[
            'away_team', 'home_rating'
        ]].rename(columns={
            'away_team' : 'team',
            'home_rating' : 'opponent_rating'
        })
    ])
    ## group by team and calc sos ##
    sos_df = flat.groupby(['team']).agg(
        sos = ('opponent_rating', 'mean'),
    ).reset_index()
    return sos_df


def optimize_season(df, games, season):
    print('          Optimizing {0} season...'.format(season))
    ## filter dfs ##
    lines_ = df[
        df['season'] == season
    ].copy()
    ## games ##
    games_ = games[
        (games['season'] == season) &
        (games['game_type'] == 'REG')
    ].copy()
    ## get params ##
    best_guesses, var_keys, bounds_list = construct_vars(lines_)
    bounds = tuple(bounds_list)
    ## optimize ##
    solution = minimize(
        obj_func,
        best_guesses,
        args=(
            lines_,
            games_,
            var_keys
        ),
        bounds=bounds,
        method='SLSQP'
    )
    ## do same for elo based ##
    best_guesses_elo, var_keys_elo, bounds_list_elo = construct_vars_elo(lines_)
    bounds_elo = tuple(bounds_list_elo)
    ## optimize ##
    solution_elo = minimize(
        obj_func_elo,
        best_guesses_elo,
        args=(
            lines_,
            games_,
            var_keys_elo
        ),
        bounds=bounds_elo,
        method='SLSQP'
    )
    ## output ##
    out_data = []
    for team in var_keys:
        if team == 'HFA':
            pass
        else:
            out_data.append({
                'season' : season,
                'team' : team,
                'wt_rating' : solution.x[var_keys.index(team)],
                'wt_rating_elo' : solution_elo.x[var_keys_elo.index(team)]
            })
    out = pd.DataFrame(out_data)
    ## add sos ##
    sos = calc_sos(solution.x, var_keys, games_)
    out = pd.merge(
        out,
        sos,
        on=['team'],
        how='left'
    )
    return out


def update_wt_ratings(force=False):
    ## final wrapper function that calls all the above ##
    print('Updating WT Ratings...')
    ## load data ##
    print('     Loading data...')
    totals, games, logos, existing = load_dfs()
    ## make line adjustments ##
    totals = calc_vig_free_odds(totals, wt_config)
    ## define update logic ##
    start = totals['season'].min()
    if existing is not None and not force:
        start = existing['season'].max()
    end = games['season'].max()
    update = False
    if start < end:
        update = True
    ## update ##
    if update:
        print('     Calculating from {0} to to {1}'.format(
            start, end
        ))
        new_dfs = []
        for season in range(start, end +1):
            rtg = optimize_season(totals, games, season)
            new_dfs.append(rtg)
        ## combine ##
        new = pd.concat(new_dfs)
        ## add back lines ##
        print('     Formatting and saving...')
        new = pd.merge(
            new,
            totals[[
                'season', 'team', 'line',
                'over_odds', 'under_odds', 'line_adj'
            ]],
            on=['season', 'team'],
            how='left'
        )
        ## add logos ##
        new = pd.merge(
            new,
            logos[[
                'team_abbr','team_nick','team_color','team_color2',
                'team_logo_espn'
            ]].rename(columns={
                'team_abbr' : 'team'
            }),
            on=['team'],
            how='left'
        )
        ## merge with existing ##
        if existing is None or force:
            existing = new
        else:
            ## remove newly pulled ##
            existing = existing[
                ~numpy.isin(
                    existing['season'],
                    new['season'].unique().tolist()
                )
            ].copy()
            ## add new ##
            existing = pd.concat([
                existing,
                new
            ])
        existing = existing.sort_values(
            by=['season','team'],
            ascending=[True, True]
        ).reset_index(drop=True)
        ## apply probs and holds ##
        existing['over_probability'], existing['under_probability'], existing['hold'] = calc_probs_and_hold(
            existing['over_odds'],
            existing['under_odds']
        )
        ## save ##
        existing.to_csv(
            '{0}/{1}/wt_ratings.csv'.format(
                package_dir,
                output_folder
            )
        )
        print('     Done!')
