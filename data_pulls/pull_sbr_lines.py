## packages ##
import pandas as pd
import numpy
import datetime
import pathlib
import json
import time
import random
import requests


## get package directory ##
package_dir = pathlib.Path(__file__).parent.parent.resolve()

## load config file ##
config = None
with open('{0}/config.json'.format(package_dir), 'r') as fp:
    config = json.load(fp)


api_url = config['data_pulls']['sbr']['api_url']
game_loc = config['data_pulls']['sbr']['game_loc']
season_start = config['data_pulls']['sbr']['season_start']
sbr_team_dict = config['data_pulls']['sbr']['sbr_team_dict']
week_ranges = config['data_pulls']['sbr']['week_ranges']
headers = config['data_pulls']['sbr']['headers']


## create output sub path ##
output_folder = '/data_sources/sbr'

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


## helper to establish existing data ##
def establish_existing_data():
    print('     Looking for most recent SBR line data...')
    try:
        most_recent_event_df = pd.read_csv('{0}{1}/sbr_events.csv'.format(package_dir, output_folder), index_col=0)
        most_recent_line_df = pd.read_csv('{0}{1}/sbr_lines.csv'.format(package_dir, output_folder), index_col=0)
        most_recent_event_df = most_recent_event_df[~pd.isnull(most_recent_event_df['eid'])].copy()
        most_recent_season = int(most_recent_event_df['season'].max())
        existing_event_ids = most_recent_event_df['eid'].unique()
        print('        Found data for the {0} season...'.format(most_recent_season))
    except:
        most_recent_event_df = None
        most_recent_line_df = None
        most_recent_season = season_start
        existing_event_ids = []
        print('        Didn''t find data. Rebuilding from the {0} season...'.format(season_start))
    return [
        most_recent_event_df,
        most_recent_line_df,
        most_recent_season,
        existing_event_ids
    ]


def pull_sbr_events(most_recent_event_df, most_recent_season, existing_event_ids):
    print('     Pulling event ids...')
    event_data = []
    save_update = True
    ## pull ##
    for year in range(most_recent_season, season_end +1):
        ## pause ##
        print('          Pulling the {0} season...'.format(int(year)))
        time.sleep(5 + random.random() * 5)
        ## create start time unix timestamp
        start_date = datetime.datetime.strptime('{0}-09-03'.format(year), '%Y-%m-%d')
        start_date_unix = int(datetime.datetime.timestamp(start_date)*1000)
        ## format query string ##
        ## OLD CODE ON A DEPRECATED END POINT ##
        # event_query_string = """{ eventsByDateByLeagueGroup( es: ["in-progress", "scheduled", "complete", "suspended", "delayed", "postponed", "retired", "canceled"], leagueGroups: [{ mtid: 401, lid: 16, spid: 4 }], providerAcountOpener: 8, hoursRange: 5000, showEmptyEvents: false, marketTypeLayout: "PARTICIPANTS", ic: false, startDate: $START_DATE, timezoneOffset: -4, nof: true, hl: true, sort: {by: ["lid", "dt", "des"], order: ASC} ) { events { eid lid spid des dt es rid ic ven tvs cit cou st sta hl seid writeingame plays(pgid: 2, limitLastSeq: 3, pgidWhenFinished: -1) { eid sqid siid gid nam val tim } scores { partid val eid pn sequence } participants { eid partid partbeid psid ih rot tr sppil sppic startingPitcher { fn lnam } source { ... on Player { pid fn lnam } ... on Team { tmid lid tmblid nam nn sn abbr cit senam imageurl } ... on ParticipantGroup { partgid nam lid participants { eid partid psid ih rot source { ... on Player { pid fn lnam } ... on Team { tmid lid nam nn sn abbr cit } } } } } } marketTypes { mtid spid nam des settings { sitid did alias format template sort url } }  eventGroup { egid nam } statistics(sgid: 3, sgidWhenFinished: 4) { val eid nam partid pid typ siid sequence } league { lid nam rid spid sn settings { alias rotation ord shortnamebreakpoint matchupline } } } maxSequences { events: eventsMaxSequence scores: scoresMaxSequence currentLines: linesMaxSequence statistics: statisticsMaxSequence plays: playsMaxSequence consensus: consensusMaxSequence } } }""".replace('$START_DATE', '{0}'.format(start_date_unix))
        # ## format request ##
        # payload = {
        #     'query' : event_query_string
        # }
        # ## make request ##
        # r = requests.get(
        #     api_url,
        #     params=payload,
        #     headers=headers
        # )
        # if r.status_code != 200:
        #     print('          Request failed...')
        #     print('          Status code: {0}'.format(r.status_code))
        #     print('          Text: {0}'.format(r.text))
        # ## parse response
        # r_json = r.json()
        # try:
        #     events = r_json['data']['eventsByDateByLeagueGroup']['events']
        # except Exception as e:
        #     print('          Error parsing json...')
        #     print('          Error: {0}'.format(e))
        #     print('          JSON: {0}'.format(r_json))
        # events = r_json['data']['eventsByDateByLeagueGroup']['events']
        ## format query string ##
        event_query_string = '''
        {eventsByDateNew(
          startDate: $START_DATE,
          hoursRange: 5000,
          lid: [16],
          mtid: [401],
          showEmptyEvents: false,
          sort: {
              by: ["lid", "dt", "des"],
              order: ASC
          },
          es: ["in-progress", "scheduled", "complete", "suspended", "delayed", "postponed", "retired", "canceled"]
        ) { events { eid lid spid des dt es rid ic ven tvs cit cou st sta hl seid writeingame plays(pgid: 2, limitLastSeq: 3, pgidWhenFinished: -1) { eid sqid siid gid nam val tim } scores { partid val eid pn sequence } participants { eid partid partbeid psid ih rot tr sppil sppic startingPitcher { fn lnam } source { ... on Player { pid fn lnam } ... on Team { tmid lid tmblid nam nn sn abbr cit senam imageurl } ... on ParticipantGroup { partgid nam lid participants { eid partid psid ih rot source { ... on Player { pid fn lnam } ... on Team { tmid lid nam nn sn abbr cit } } } } } } marketTypes { mtid spid nam des settings { sitid did alias format template sort url } }  eventGroup { egid nam } statistics(sgid: 3, sgidWhenFinished: 4) { val eid nam partid pid typ siid sequence } league { lid nam rid spid sn settings { alias rotation ord shortnamebreakpoint matchupline } } } maxSequences { events: eventsMaxSequence scores: scoresMaxSequence currentLines: linesMaxSequence statistics: statisticsMaxSequence plays: playsMaxSequence consensus: consensusMaxSequence } } }
        '''.replace('$START_DATE', '{0}'.format(start_date_unix))
        ## format request ##
        payload = {
            'query' : event_query_string
        }
        ## make request ##
        r = requests.get(
            api_url,
            params=payload,
            headers=headers
        )
        if r.status_code != 200:
            print('          Request failed...')
            print('          Status code: {0}'.format(r.status_code))
            print('          Text: {0}'.format(r.text))
        ## parse response
        r_json = r.json()
        try:
            events = r_json['data']['eventsByDateNew']['events']
        except Exception as e:
            print('          Error parsing json...')
            print('          Error: {0}'.format(e))
            print('          JSON: {0}'.format(r_json))
        events = r_json['data']['eventsByDateNew']['events']
        ## add data to container ##
        for event in events:
            try:
                ## determine home and away ##
                at_tmid = None
                at_abbr = None
                ht_tmid = None
                ht_abbr = None
                at_rot = None
                ht_rot = None
                if int(event['participants'][0]['rot']) > int(event['participants'][1]['rot']):
                    at_tmid = event['participants'][1]['source']['tmid']
                    at_abbr = event['participants'][1]['source']['abbr']
                    ht_tmid = event['participants'][0]['source']['tmid']
                    ht_abbr = event['participants'][0]['source']['abbr']
                    at_rot = event['participants'][1]['rot']
                    ht_rot = event['participants'][0]['rot']
                else:
                    ht_tmid = event['participants'][1]['source']['tmid']
                    ht_abbr = event['participants'][1]['source']['abbr']
                    at_tmid = event['participants'][0]['source']['tmid']
                    at_abbr = event['participants'][0]['source']['abbr']
                    at_rot = event['participants'][0]['rot']
                    ht_rot = event['participants'][1]['rot']
                ## determine scores ##
                home_1h = 0
                home_2h = 0
                home_fg = 0
                away_1h = 0
                away_2h = 0
                away_fg = 0
                for s in event['scores']:
                    if s['pn'] in [1,2]:
                        if s['partid'] == ht_tmid:
                            home_1h += int(s['val'])
                            home_fg += int(s['val'])
                        else:
                            away_1h += int(s['val'])
                            away_fg += int(s['val'])
                    elif s['pn'] in [3,4]:
                        if s['partid'] == ht_tmid:
                            home_2h += int(s['val'])
                            home_fg += int(s['val'])
                        else:
                            away_2h += int(s['val'])
                            away_fg += int(s['val'])
                    else:
                        if s['partid'] == ht_tmid:
                            home_fg += int(s['val'])
                        else:
                            away_fg += int(s['val'])
                event_data.append({
                    'season' : year,
                    'dt' : event['dt'],
                    'eid' : event['eid'],
                    'des' : event['des'],
                    'ven' : event['ven'],
                    'cit' : event['cit'],
                    'sta' : event['sta'],
                    'cou' : event['cou'],
                    'st' : event['st'],
                    'at_tmid' : at_tmid,
                    'at_abbr' : at_abbr,
                    'at_rot' : at_rot,
                    'ht_tmid' : ht_tmid,
                    'ht_abbr' : ht_abbr,
                    'ht_rot' : ht_rot,
                    'ht_1h_score' : home_1h,
                    'ht_2h_score' : home_2h,
                    'ht_fg_score' : home_fg,
                    'at_1h_score' : away_1h,
                    'at_2h_score' : away_2h,
                    'at_fg_score' : away_fg,
                })
            except Exception as e:
                ## special catch for games with bad events ##
                if event['eid'] in [73]:
                    event_data.append({
                        'season' : year,
                        'dt' : 0,
                        'eid' : event['eid'],
                        'des' : event['des'],
                        'ven' : event['ven'],
                        'cit' : event['cit'],
                        'sta' : event['sta'],
                        'cou' : event['cou'],
                        'st' : event['st'],
                        'at_tmid' : 1529,
                        'at_abbr' : 'JAC',
                        'at_rot' : numpy.nan,
                        'ht_tmid' : 1527,
                        'ht_abbr' : 'IND',
                        'ht_rot' : numpy.nan,
                        'ht_1h_score' : numpy.nan,
                        'ht_2h_score' : numpy.nan,
                        'ht_fg_score' : numpy.nan,
                        'at_1h_score' : numpy.nan,
                        'at_2h_score' : numpy.nan,
                        'at_fg_score' : numpy.nan,
                    })
                else:
                    print('               Did not parse {0}'.format(
                        event['des']
                    ))
                    print('                    Error: {0}'.format(
                        e
                    ))
    event_df = pd.DataFrame(event_data)
    ## load fastr game data for matching game_id ##
    games_df = pd.read_csv(
        '{0}/{1}'.format(
            package_dir,
            game_loc
        ),
        index_col=0
    )
    ## standardize names ##
    event_df['home_team'] = event_df['ht_abbr'].replace(sbr_team_dict)
    event_df['away_team'] = event_df['at_abbr'].replace(sbr_team_dict)
    ## filer down game df for game matching ##
    ## cant use post season with this logic ##
    games_df = games_df[
        (games_df['game_type'] == 'REG') &
        (games_df['season'] >= most_recent_season) &
        (games_df['season'] <= season_end)
    ].copy()
    ## merge ##
    merged_df = pd.merge(
        games_df[['season', 'home_team', 'away_team', 'game_id', 'week']],
        event_df,
        on=['season', 'home_team', 'away_team'],
        how='left'
    )
    ## remove dupes from playoffs ##
    merged_df = merged_df.sort_values(
        by=['dt'],
        ascending=[True]
    ).reset_index(drop=True)
    merged_df = merged_df.groupby(['game_id']).head(1)
    ## merge and save ##
    updated_eids = merged_df['eid'].unique().tolist()
    if most_recent_event_df is None:
        event_df = merged_df.copy()
    else:
        most_recent_event_df = most_recent_event_df[
            ~numpy.isin(
                most_recent_event_df['eid'],
                updated_eids
            )
        ]
        event_df = pd.concat([
            most_recent_event_df,
            merged_df
        ]).copy()
    ## save ##
    event_df.to_csv(
        '{0}{1}/sbr_events.csv'.format(
            package_dir,
            output_folder
        )
    )
    return event_df



def update_sbr_lines(most_recent_event_df, most_recent_line_df):
    print('     Pulling lines...')
    ## establish start date for line pull ##
    if most_recent_line_df is None:
        season_start_lines = season_start
    else:
        season_start_lines = int(most_recent_line_df['season'].max())
    line_data = []
    save_update = True
    ## pull ##
    for year in range(season_start_lines, season_end +1):
        ## pause ##
        print('          Pulling the {0} season...'.format(int(year)))
        for week_range in week_ranges:
            print('               On weeks {0} through {1}'.format(
                min(week_range),
                max(week_range)
            ))
            time.sleep(5 + random.random() * 15)
            ## get eids in season and week range ##
            eids = most_recent_event_df[
                (most_recent_event_df['season'] == year) &
                (~pd.isnull(most_recent_event_df['eid'])) &
                numpy.isin(
                    most_recent_event_df['week'],
                    week_range
                )
            ]['eid'].tolist()
            if len(eids) == 0:
                print('               No events found. Skippnig...')
            else:
                ## add to query string ##
                line_query_string = """ { currentLines(eid: [$EIDS], mtid: [401], marketTypeLayout: "PARTICIPANTS", paid: 8) openingLines(eid: [$EIDS], mtid: [401], marketTypeLayout: "PARTICIPANTS", paid: 8) }""".replace(
                    '$EIDS', ', '.join(str(int(eid)) for eid in eids)
                )
                ## format request ##
                payload = {
                    'query' : line_query_string
                }
                ## make request ##
                tries = 1
                success = False
                r = None
                while tries <= 5 and not success:
                    r = requests.get(
                        api_url,
                        params=payload,
                        headers=headers
                    )
                    if r.status_code == 200:
                        success = True
                    else:
                        print('                    Did not receive valid response. Pausing and retrying...')
                        time.sleep(5 + random.random() * 5 * tries)
                        tries += 1
                if not success:
                    print('                    Could not get a valid response, halting script')
                    break
                else:
                    ## parse response
                    r_json = r.json()
                    closing_lines = r_json['data']['currentLines']
                    opening_lines = r_json['data']['openingLines']
                    ## add data to container ##
                    for line in closing_lines:
                        try:
                            line_data.append({
                                'eid' : line['eid'],
                                'mtid' : line['mtid'],
                                'paid' : line['paid'],
                                'partid' : line['partid'],
                                'adj' : line['adj'],
                                'pri' : line['pri'],
                                'ap' : line['ap'],
                                'tim' : line['tim'],
                                'type' : 'closing',
                                'season' : year,
                            })
                        except:
                            print('               Did not parse {0}'.format(
                                line
                            ))
                    ## add data to container ##
                    for line in opening_lines:
                        try:
                            line_data.append({
                                'eid' : line['eid'],
                                'mtid' : line['mtid'],
                                'paid' : line['paid'],
                                'partid' : line['partid'],
                                'adj' : line['adj'],
                                'pri' : line['pri'],
                                'ap' : line['ap'],
                                'tim' : line['tim'],
                                'type' : 'opening',
                                'season' : year,
                            })
                        except:
                            print('               Did not parse {0}'.format(
                                line
                            ))
    ## translate to dataframe ##
    lines_df = pd.DataFrame(line_data)
    ## merge and save ##
    updated_eids = lines_df['eid'].unique().tolist()
    if most_recent_line_df is None:
        updated_line_df = lines_df.copy()
    else:
        most_recent_line_df = most_recent_line_df[
            ~numpy.isin(
                most_recent_line_df['eid'],
                updated_eids
            )
        ]
        updated_line_df = pd.concat([
            most_recent_line_df,
            lines_df
        ]).copy()
    ## save ##
    updated_line_df.to_csv(
        '{0}{1}/sbr_lines.csv'.format(
            package_dir,
            output_folder
        )
    )
    return updated_line_df





def pull_sbr_lines():
    print('Pulling SBR data...')
    ## establish existing events ##
    existing_data = establish_existing_data()
    ## pull events ##
    event_df = pull_sbr_events(
        existing_data[0],
        existing_data[2],
        existing_data[3]
    )
    ## pull lines ##
    line_df = update_sbr_lines(
        event_df,
        existing_data[1]
    )
    print('     Merging events and lines...')
    ## merge events and lines ##
    ## home close ##
    output_df = pd.merge(
        event_df,
        line_df[
            line_df['type'] == 'closing'
        ][[
            'eid', 'partid', 'adj', 'ap'
        ]].rename(columns={
            'partid' : 'ht_tmid',
            'adj' : 'sbr_home_closing_line',
            'ap' : 'sbr_home_closing_price',
        }),
        on=['eid', 'ht_tmid'],
        how='left'
    )
    ## home open ##
    output_df = pd.merge(
        output_df,
        line_df[
            line_df['type'] == 'opening'
        ][[
            'eid', 'partid', 'adj', 'ap'
        ]].rename(columns={
            'partid' : 'ht_tmid',
            'adj' : 'sbr_home_opening_line',
            'ap' : 'sbr_home_opening_price',
        }),
        on=['eid', 'ht_tmid'],
        how='left'
    )
    ## close close ##
    output_df = pd.merge(
        output_df,
        line_df[
            line_df['type'] == 'closing'
        ][[
            'eid', 'partid', 'adj', 'ap'
        ]].rename(columns={
            'partid' : 'at_tmid',
            'adj' : 'sbr_away_closing_line',
            'ap' : 'sbr_away_closing_price',
        }),
        on=['eid', 'at_tmid'],
        how='left'
    )
    ## close open ##
    output_df = pd.merge(
        output_df,
        line_df[
            line_df['type'] == 'opening'
        ][[
            'eid', 'partid', 'adj', 'ap'
        ]].rename(columns={
            'partid' : 'at_tmid',
            'adj' : 'sbr_away_opening_line',
            'ap' : 'sbr_away_opening_price',
        }),
        on=['eid', 'at_tmid'],
        how='left'
    )
    ## save ##
    output_df.to_csv(
        '{0}{1}/sbr_combined.csv'.format(
            package_dir,
            output_folder
        )
    )
