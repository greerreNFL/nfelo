## packages ##
import pandas as pd
import numpy
import datetime
import pathlib
import json
import random
import time
import requests
from bs4 import BeautifulSoup


## get package directory ##
package_dir = pathlib.Path(__file__).parent.parent.resolve()

## load config file ##
config = None
with open('{0}/config.json'.format(package_dir), 'r') as fp:
    config = json.load(fp)


## scraping params ##
headers = config['data_pulls']['tfl']['headers']
col_headers = config['data_pulls']['tfl']['col_headers']
tfl_team_dict = config['data_pulls']['tfl']['tfl_team_dict']


## create output sub path ##
output_folder = '/data_sources/tfl'


def scrape_spreads():
    print('     Scraping spreads...')
    data_rows = []
    urls = []
    ## determine regular season weeks to scrape ##
    ## if past 2021, look for a week 18 ##
    if int(datetime.datetime.today().year) >= 2022:
        for rw in range(1,19):
            urls.append(
                'https://thefootballlines.com/nfl/week-{0}/point-spreads'.format(
                    rw
                )
            )
    else:
        for rw in range(1,18):
            urls.append(
                'https://thefootballlines.com/nfl/week-{0}/point-spreads'.format(
                    rw
                )
            )
    ## add playoffs ##
    for pw in ['wildcard','division','conference','superbowl']:
        urls.append(
            'https://thefootballlines.com/nfl/{0}/point-spreads'.format(
                pw
            )
        )
    for week in urls:
        print('          On week {0} of {1}...'.format(
            urls.index(week) + 1,len(urls)
        ))
        time.sleep((2.5 + random.random() * 5))
        try:
            raw = requests.get(
                week,
                headers=headers
            )
        except:
            print('               Couldnt get URL. Pausing for 10 seconds and retrying... ')
            time.sleep(10)
            try:
                raw = requests.get(
                    week,
                    headers=headers
                )
            except:
                print('               Couldnt get URL again...')
                print('               Skipping Week...')
                continue
        parsed = BeautifulSoup(raw.content, "html.parser")
        trs_odd = parsed.find_all('tr',{'class' : 'odd'})
        trs_even = trs = parsed.find_all('tr',{'class' : 'even'})
        trs_all = []
        for tr in trs_odd:
            trs_all.append(tr)
        for tr in trs_even:
            trs_all.append(tr)
        for tr in trs_all:
            tds_unf = tr.find_all('td')
            tds = []
            for td in tds_unf:
                stringed = str(td.text) ##convert from unicode##
                stringed = stringed.replace('\n','') ## replace line breaks with blanks##
                stringed = stringed.strip() ## remove spaces ##
                tds.append(stringed)
            away_team = tds[0].split(' @ ')[0].split(' ')[0]
            home_team = tds[0].split(' @ ')[1].split(' ')[0]
            away_score = tds[0].split(' @ ')[0].split(' ')[1]
            home_score = tds[0].split(' @ ')[1].split(' ')[1]
            if int(tds[1][5:7]) < 7:
                season = int(tds[1][:4]) - 1
            else:
                season = int(tds[1][:4])
            if 'week' in week:
                week_num = int(week.split('/week-')[1].split('/')[0])
            else:
                if int(datetime.datetime.today().year) >= 2022:
                    week_num = 19
                else:
                    week_num = 18
            opening_home_line = float(tds[5])
            closing_home_line = float(tds[6])
            row_data = {
                'season' : season,
                'week' : week_num,
                'home_team' : home_team,
                'away_team' : away_team,
                'home_score' : home_score,
                'away_score' : away_score,
                'opening_home_line_tfl' : opening_home_line,
                'closing_home_line_tfl' : closing_home_line,
            }
            data_rows.append(row_data)
    return pd.DataFrame(data_rows)


def scrape_mls():
    print('     Scraping moneylines...')
    data_rows_ml = []
    urls_ml = []
    ## determine regular season weeks to scrape ##
    ## if past 2021, look for a week 18 ##
    if int(datetime.datetime.today().year) >= 2022:
        for rw in range(1,19):
            urls_ml.append(
                'https://thefootballlines.com/nfl-odds/week-{0}'.format(
                    rw
                )
            )
    else:
        for rw in range(1,18):
            urls_ml.append(
                'https://thefootballlines.com/nfl-odds/week-{0}'.format(
                    rw
                )
            )
    ## add playoffs ##
    for pw in ['wildcard','division','conference','superbowl']:
        urls_ml.append(
            'https://thefootballlines.com/nfl-odds/{0}'.format(
                pw
            )
        )
    for week in urls_ml:
        print('          On week {0} of {1}...'.format(
            urls_ml.index(week) + 1,len(urls_ml)
        ))
        time.sleep((2.5 + random.random() * 5))
        try:
            raw = requests.get(
                week,
                headers=headers
            )
        except:
            print('               Couldnt get URL. Pausing for 10 seconds and retrying... ')
            time.sleep(10)
            try:
                raw = requests.get(
                    week,
                    headers=headers
                )
            except:
                print('               Couldnt get URL again...')
                print('               Skipping Week...')
                continue
        parsed = BeautifulSoup(raw.content, "html.parser")
        trs_odd = parsed.find_all('tr',{'class' : 'odd'})
        trs_even = trs = parsed.find_all('tr',{'class' : 'even'})
        trs_all = []
        for tr in trs_odd:
            trs_all.append(tr)
        for tr in trs_even:
            trs_all.append(tr)
        for tr in trs_all:
            tds_unf = tr.find_all('td')
            tds = []
            for td in tds_unf:
                stringed = str(td.text) ##convert from unicode##
                stringed = stringed.replace('\n','') ## replace line breaks with blanks##
                stringed = stringed.strip() ## remove spaces ##
                tds.append(stringed)
            away_team = tds[0].split(' @ ')[0].split(' ')[0]
            home_team = tds[0].split(' @ ')[1].split(' ')[0]
            away_score = tds[0].split(' @ ')[0].split(' ')[1]
            home_score = tds[0].split(' @ ')[1].split(' ')[1]
            if int(tds[1][5:7]) < 7:
                season = int(tds[1][:4]) - 1
            else:
                season = int(tds[1][:4])
            if 'week' in week:
                week_num = int(week.split('/week-')[1].split('/')[0])
            else:
                week_num = 18
            opening_away_money_line = float(tds[2].split(' ')[0])
            closing_away_money_line = float(tds[3].split(' ')[0])
            opening_home_money_line = float(tds[4].split(' ')[0])
            closing_home_money_line = float(tds[5].split(' ')[0])
            row_data = {
                'season' : season,
                'week' : week_num,
                'home_team' : home_team,
                'away_team' : away_team,
                'opening_away_money_line_tfl' : opening_away_money_line,
                'closing_away_money_line_tfl' : closing_away_money_line,
                'opening_home_money_line_tfl' : opening_home_money_line,
                'closing_home_money_line_tfl' : closing_home_money_line,
            }
            data_rows_ml.append(row_data)
    return pd.DataFrame(data_rows_ml)




def pull_tfl_lines():
    print('Scraping lines from TFL -- TFL is a primary source for consensus openers...')
    ## get spreads ##
    unformatted_df = scrape_spreads()
    ## get mls ##
    unformatted_ml_df = scrape_mls()
    ## merge and format ##
    print('     Merging and formatting...')
    ## rename team abbrs ##
    unformatted_df['home_team'] = unformatted_df['home_team'].replace(tfl_team_dict)
    unformatted_df['away_team'] = unformatted_df['away_team'].replace(tfl_team_dict)
    unformatted_ml_df['home_team'] = unformatted_ml_df['home_team'].replace(tfl_team_dict)
    unformatted_ml_df['away_team'] = unformatted_ml_df['away_team'].replace(tfl_team_dict)
    ## merge ##
    merged_unformatted_df = pd.merge(
        unformatted_df,
        unformatted_ml_df,
        on=['season','week','home_team','away_team'],
        how='left'
    )
    ## filter columns ##
    formatted_df = merged_unformatted_df[col_headers]
    ## sort ##
    formatted_df = formatted_df.sort_values(
        by=['season','week']
    ).reset_index(drop=True)
    ## export ##
    formatted_df.to_csv(
        '{0}{1}/tfl_lines.csv'.format(
            package_dir,
            output_folder
        )
    )
    print('     Done!')
