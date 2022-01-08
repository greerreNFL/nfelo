## packages ##
import pandas as pd
import numpy
import datetime
import pathlib
import json
import random
import time
import requests

## get package directory ##
package_dir = pathlib.Path(__file__).parent.parent.resolve()

## load config file ##
config = None
with open('{0}/config.json'.format(package_dir), 'r') as fp:
    config = json.load(fp)

config_private = None
with open('{0}/config_private.json'.format(package_dir), 'r') as fp:
    config_private = json.load(fp)


## scraping params ##
raw_cookie_text = config_private['pff']['raw_cookie_text']
headers = config['data_pulls']['pff']['headers']
referer_team_dict = config['data_pulls']['pff']['referer_team_dict']
season_start = config['data_pulls']['pff']['starting_season']

## formatting params ##
pff_franchise_to_standard_dict = config['data_pulls']['pff']['pff_standardization']
home_column_translation = config['data_pulls']['pff']['home_column_translation']
away_column_translation = config['data_pulls']['pff']['away_column_translation']
pff_headers = config['data_pulls']['pff']['final_columns']


## create output sub path ##
output_folder = '/data_sources/pff'

## establish range for downloads ##
season_end = None

## set max season ##
if datetime.date.today() > datetime.date(
    year=datetime.date.today().year,
    month=9,
    day=6
):
    season_end = datetime.date.today().year
else:
    season_end = datetime.date.today().year - 1


## translate raw cookie text into dict ##
cookie_dict = {}
cookie_list = raw_cookie_text.split(';')
try:
    for i in cookie_list:
        cookie_dict[i.split('=')[0]] = i.split('=')[1]
except:
    print('   ************************* WARNING *************************')
    print('   *** PFF cookie did not load. Likely needs to be updated ***')


## rewrite config to file to match whats on the github repo, so the tracked file
## remains unchanged and updates can be pushed ##
config_private['pff']['raw_cookie_text'] = "YOUR COOKIE TEXT HERE"
with open('{0}/config_private.json'.format(package_dir), 'w') as fp:
    json.dump(config_private, fp, indent=2)




## helper to establish existing data ##
def establish_existing_data():
    print('     Looking for most recent pff game data...')
    try:
        most_recent_pff_game_df = pd.read_csv('{0}{1}/pff_game.csv'.format(package_dir, output_folder), index_col=0)
        most_recent_pff_flat_df = pd.read_csv('{0}{1}/pff_game_flat.csv'.format(package_dir, output_folder), index_col=0)
        most_recent_season = int(most_recent_pff_game_df['season'].max())
        existing_game_ids = most_recent_pff_game_df['game_id'].unique()
        print('        Found data for the {0} season...'.format(most_recent_season))
    except:
        most_recent_pff_game_df = None
        most_recent_pff_flat_df = None
        most_recent_season = season_start
        existing_game_ids = []
        print('        Didn''t find data. Rebuilding from the {0} season...'.format(season_start))
    return [most_recent_pff_game_df, most_recent_pff_flat_df, most_recent_season, existing_game_ids]

## helper that pulls the data ##
def data_pull():
    print('Updating PFF game grade data...')
    ## struct for scraped data ##
    pff_output = []
    ## flag to avoid updating impartial data pull ##
    save_update = True
    ## establish current data ##
    existing_data = establish_existing_data()
    most_recent_pff_game_df = existing_data[0]
    most_recent_pff_flat_df = existing_data[1]
    most_recent_season = existing_data[2]
    existing_game_ids = existing_data[3]
    ## pull ##
    while most_recent_season <= season_end:
        print('        Pulling {0} pff game grades...'.format(most_recent_season))
        for team in range(1,33):
            print('             On team {0} of 32...'.format(team))
            time.sleep((5 + random.random() * 25)) ## be kind to their servers ##
            payload = {
                'league' : 'nfl',
                'season' : str(most_recent_season),
                'week' : '1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21',
                'franchise_id' : str(team),
            }
            url = 'https://premium.pff.com/api/v1/teams/summary?'
            ## copy headers to add a refer param ##
            temp_headers = headers.copy()
            temp_headers['referer'] = 'https://premium.pff.com/nfl/teams/{0}/summary?season={1}&weekGroup=REG'.format(
                referer_team_dict[str(team)],
                most_recent_season
            )
            ## make request ##
            raw = requests.get(
                url,
                cookies=cookie_dict,
                params=payload,
                headers=temp_headers
            )
            ## check for bad response ##
            if raw.status_code != 200:
                print('                  Got a bad response. Updated cookie and rerun')
                save_update = False
                break
            else:
                pass
            json_data = json.loads(raw.content)
            resp_keys = []
            ## parse response ##
            try:
                if len(json_data['team_summary']) == 0:
                    pass
                else:
                    for game in json_data['team_summary']:
                        if game['lock_status'] == 'processed': ## skip ungraded games ##
                            if game['game_id'] in existing_game_ids: ## drop old scores that have been updated ##
                                most_recent_pff_game_df = most_recent_pff_game_df[most_recent_pff_game_df['game_id']!=game['game_id']]
                                most_recent_pff_flat_df = most_recent_pff_flat_df[most_recent_pff_flat_df['game_id']!=game['game_id']]
                            else:
                                pass
                            row_data = {}
                            missing_count = 0
                            for meta in [
                                'game_id', 'week', 'start', 'franchise_id',
                                'opponent_franchise_id', 'home', 'points_scored', 'points_allowed',
                                'grades_coverage_defense', 'grades_defense', 'grades_misc_st',
                                'grades_offense', 'grades_overall', 'grades_pass', 'grades_pass_block',
                                'grades_pass_route', 'grades_pass_rush_defense', 'grades_run',
                                'grades_run_block', 'grades_run_defense'
                            ]:
                                if 'grades_' in meta:
                                    meta_name = meta.split('grades_')[1]
                                else:
                                    meta_name = meta
                                try:
                                    row_data[meta_name] = game[meta]
                                except:
                                    print('                  Couldnt find {0}'.format(meta))
                                    row_data[meta_name] = numpy.nan
                                    missing_count += 1
                            if missing_count > 5:
                                print('                  Too many metrics missing. Somethign is wrong! Check cookie and parser')
                                save_update = False
                            row_data['season'] = most_recent_season
                            pff_output.append(row_data)
                        else:
                            continue
            except Exception as e:
                print('                  Could not parse response. Updated cookie and rerun')
                print('                       Error: {0}'.format(e))
                print('                       If error persists, script may need to updated!')
                save_update = False
                break
        if save_update:
            most_recent_season += 1
        else:
            most_recent_season += 9999
    ## if data was susccessfully parse, format and update ##
    if save_update:
        ## create update df ##
        pff_df = pd.DataFrame(pff_output)
        pff_df.to_csv('{0}{1}/temp.csv'.format(
            package_dir,
            output_folder
        ))
        ## turn franchise ids to str for repl ##
        pff_df['franchise_id'] = pff_df['franchise_id'].astype('str')
        pff_df['franchise_id'] = pff_df['franchise_id'].replace(pff_franchise_to_standard_dict)
        pff_df['opponent_franchise_id'] = pff_df['opponent_franchise_id'].astype('str')
        pff_df['opponent_franchise_id'] = pff_df['opponent_franchise_id'].replace(pff_franchise_to_standard_dict)
        print('     Compiling data...')
        if most_recent_pff_game_df is None:
            pff_df_join = pff_df
        else:
            pff_df_join = pd.concat([most_recent_pff_flat_df,pff_df])
            pff_df_join = pff_df_join.drop_duplicates()
        pff_df_join = pff_df_join.sort_values(by=['season','franchise_id']).reset_index(drop=True)
        ## deflatten file
        home_flat = pff_df_join[pff_df_join['home']].rename(
            columns=home_column_translation
        ).drop(columns=['start','opponent_franchise_id','home'])
        away_flat = pff_df_join[~pff_df_join['home']].rename(
            columns=away_column_translation
        ).drop(columns=['start','opponent_franchise_id','home','points_scored','points_allowed'])
        ## merge ##
        pff_game_df = pd.merge(
            home_flat,
            away_flat,
            on=['game_id','season','week'],how='inner'
        ).sort_values(by=['game_id'])
        pff_game_df = pff_game_df[pff_headers]
        ## save files ##
        pff_df_join.to_csv('{0}{1}/pff_game_flat.csv'.format(
            package_dir,
            output_folder
        ))
        pff_game_df.to_csv('{0}{1}/pff_game.csv'.format(
            package_dir,
            output_folder
        ))
        print('     Saved!')
    else:
        pass


def pull_pff_grades():
    data_pull()
