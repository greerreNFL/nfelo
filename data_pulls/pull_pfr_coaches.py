## packages ##
import pandas as pd
import numpy
import datetime
import pathlib
import json
import time
import random
import requests
from bs4 import BeautifulSoup


## get package directory ##
package_dir = pathlib.Path(__file__).parent.parent.resolve()

## load config file ##
config = None
with open('{0}/config.json'.format(package_dir), 'r') as fp:
    config = json.load(fp)


coach_url = config['data_pulls']['pfr']['coach_url']
output_folder = '/data_sources/pfr'


def id_from_url(url):
    try:
        return url.split('.htm')[0].split('/coaches/')[1]
    except:
        return numpy.nan


def establish_existing_data():
    print('     Looking for existing coaching file...')
    try:
        coach_df = pd.read_csv(
            '{0}{1}/coaches.csv'.format(
                package_dir,
                output_folder
            ),
            index_col = 0
        )
        print('          Existing file found...')
    except:
        print('          No existing file found...')
        coach_df = None
    return coach_df


def scrape_coaches(url):
    ## container for scraped data ##
    data = []
    ## scrape pfr caoching page ##
    resp = requests.get(url)
    if resp.status_code == 200:
        soup = BeautifulSoup(resp.content, "html.parser")
        ## find coach cells ##
        coach_tds = soup.findAll('td', {'data-stat' : 'coach'})
        if len(coach_tds) > 0:
            ## scrape each coach ##
            for coach in coach_tds:
                anchor = coach.findAll('a', href=True)
                if len(anchor) > 0:
                    row = {}
                    row['pfr_coach_id'] = id_from_url(
                        anchor[0]['href']
                    )
                    row['pfr_coach_name'] = anchor[0].text
                    row['pfr_coach_url'] = anchor[0]['href']
                    data.append(row)
        else:
            print('          Did not find any coaches, might need to update scraper...')
    else:
        print('          Did not receive valid response...')
    ## handle return ##
    if len(data) > 0:
        return pd.DataFrame(data)
    else:
        return None


def scrape_coach_page(stub):
    ## container for scraped data ##
    img_url = numpy.nan
    ## scrape pfr caoching page ##
    time.sleep(5 + random.random() * 5)
    resp = None
    try:
        resp = requests.get(
            'https://www.pro-football-reference.com/{0}'.format(
                stub
            ),
            timeout=5
        )
    except:
        pass
    if resp is None:
        pass
    elif resp.status_code == 200:
        soup = BeautifulSoup(resp.content, "html.parser")
        ## find image cells ##
        image_block = soup.findAll('div', {'id' : 'meta'})
        if len(image_block) > 0:
            img = image_block[0].findAll('img')
            if len(img) > 0:
                img_url = img[0]['src']
        else:
            pass
    else:
        pass
    ## handle return ##
    return img_url


def apply_headshot(row):
    ## func for iterating through df and applying headshot ##
    headshot_url = numpy.nan
    curr_date = datetime.datetime.today().strftime('%Y-%m-%d')
    if pd.isnull(row['pfr_coach_url']):
        return row
    else:
        if pd.isnull(row['pfr_coach_image_url_last_checked']):
            ## if no record of updating, try to get hs ##
            headshot_url = scrape_coach_page(row['pfr_coach_url'])
        elif (
            datetime.datetime.strptime(
                curr_date,
                '%Y-%m-%d'
            ).date() -
            datetime.datetime.strptime(
                row['pfr_coach_image_url_last_checked'],
                '%Y-%m-%d'
            ).date()
        ).days > 365:
            ## if last update out of SLA, check ##
            headshot_url = scrape_coach_page(row['pfr_coach_url'])
        else:
            return row
    ## if we didnt return row yet, formulate for return ##
    if pd.isnull(headshot_url):
        print('          Could not find url for {0}'.format(row['pfr_coach_name']))
    else:
        print('          Updated url for {0}'.format(row['pfr_coach_name']))
    row['pfr_coach_image_url'] = headshot_url
    row['pfr_coach_image_url_last_checked'] = curr_date
    return row


def update_urls(df):
    ## iterate through the coaching df and try to add headshots ##
    print('     Updating headshots. This could take some time...')
    df = df.apply(apply_headshot, axis=1)
    return df


def merge(existing, new):
    print('     Updating coach list...')
    merged = None
    ## add pfp field to new ##
    new['pfr_coach_image_url'] = numpy.nan
    new['pfr_coach_image_url_last_checked'] = numpy.nan
    ## merge to existing
    if existing is None:
        merged = new.copy()
    else:
        existing_ids = existing['pfr_coach_id'].unique().tolist()
        new = new[
            ~numpy.isin(
                new['pfr_coach_id'],
                existing_ids
            )
        ].copy()
        merged = pd.concat([
            existing,
            new
        ])
        merged = merged.reset_index(drop=True)
    ## output ##
    return merged


## final wrapper to call all the above ##
def pull_pfr_coaches():
    ## check data ##
    existing = establish_existing_data()
    new = scrape_coaches(
        url = coach_url
    )
    merged = merge(existing, new)
    final = update_urls(merged)
    ## save ##
    final.to_csv(
        '{0}{1}/coaches.csv'.format(
            package_dir,
            output_folder
        )
    )
