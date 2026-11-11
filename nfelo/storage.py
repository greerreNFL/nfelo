## built-in ##
import os
import pathlib
import urllib.parse
from datetime import datetime, timezone

## external ##
import requests


## files written to output_data/ and published or consumed locally ##
OUTPUT_FILES = [
    'nfelo_games.csv',
    'historic_projected_spreads.csv',
    'elo_snapshot.csv',
    'prediction_tracker.csv'
]


def load_env():
    '''
    Load repo-root .env when present. Missing file is fine.
    '''
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = pathlib.Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv('.env')


def output_folder():
    '''
    Absolute path to this package's output_data directory
    '''
    return '{0}/output_data'.format(
        pathlib.Path(__file__).parent.parent.resolve()
    )


def storage_location():
    '''
    Bucket and prefix for nfelo outputs.

    Requires NFELO_STORAGE_PATH as bucket/prefix...
    '''
    load_env()
    path = os.environ.get('NFELO_STORAGE_PATH', '').strip().strip('/')
    if path == '':
        raise RuntimeError('NFELO_STORAGE_PATH is required')
    parts = path.split('/')
    if len(parts) < 2:
        raise RuntimeError(
            'NFELO_STORAGE_PATH must include a bucket and at least one prefix segment'
        )
    return parts[0], '/'.join(parts[1:])


def credentials():
    '''
    Project URL and key from the environment
    '''
    load_env()
    url = os.environ.get('STORAGE_URL', '').strip().rstrip('/')
    key = os.environ.get('STORAGE_KEY', '').strip()
    if not url or not key:
        raise RuntimeError(
            'STORAGE_URL and STORAGE_KEY are required to fetch nfelo outputs'
        )
    return url, key


def headers(key):
    return {
        'apikey': key,
        'Authorization': 'Bearer {0}'.format(key),
    }


def object_key(prefix, filename):
    return '{0}/{1}'.format(prefix, filename)


def parse_updated_at(value):
    '''
    Parse a storage updated_at string to UTC datetime
    '''
    if value is None:
        return None
    text = str(value).strip()
    if text == '':
        return None
    ## storage returns timestamps like 2026-11-11T20:00:00.000Z ##
    if text.endswith('Z'):
        text = text[:-1] + '+00:00'
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def list_output_objects():
    '''
    List nfelo output objects under the configured storage prefix.

    Returns a dict of filename -> metadata row from the storage list API.
    '''
    url, key = credentials()
    bucket, prefix = storage_location()
    response = requests.post(
        '{0}/storage/v1/object/list/{1}'.format(
            url,
            urllib.parse.quote(bucket, safe='')
        ),
        headers=headers(key),
        json={
            'prefix': '{0}/'.format(prefix),
            'limit': 100,
            'offset': 0,
        },
        timeout=(10, 60),
    )
    if response.status_code != 200:
        raise RuntimeError(
            'Failed to list nfelo storage objects ({0}): {1}'.format(
                response.status_code,
                response.text
            )
        )
    rows = response.json()
    if not isinstance(rows, list):
        raise RuntimeError('Unexpected nfelo storage list response')
    wanted = set(OUTPUT_FILES)
    found = {}
    for row in rows:
        name = row.get('name')
        if name in wanted:
            found[name] = row
    return found


def output_updated_at():
    '''
    Return filename -> updated_at datetime for each OUTPUT_FILE present
    '''
    objects = list_output_objects()
    times = {}
    for name, row in objects.items():
        times[name] = parse_updated_at(row.get('updated_at'))
    return times


def outputs_updated_since(since):
    '''
    True when every OUTPUT_FILE exists and has updated_at strictly after since
    '''
    if since.tzinfo is None:
        since = since.replace(tzinfo=timezone.utc)
    else:
        since = since.astimezone(timezone.utc)
    times = output_updated_at()
    if len(times) < len(OUTPUT_FILES):
        return False
    for name in OUTPUT_FILES:
        updated = times.get(name)
        if updated is None or updated <= since:
            return False
    return True


def download_output_file(filename):
    '''
    Download one output file bytes from storage
    '''
    if filename not in OUTPUT_FILES:
        raise ValueError('Unknown nfelo output file: {0}'.format(filename))
    url, key = credentials()
    bucket, prefix = storage_location()
    object_path = object_key(prefix, filename)
    quoted = '/'.join(
        urllib.parse.quote(segment, safe='')
        for segment in object_path.split('/')
    )
    response = requests.get(
        '{0}/storage/v1/object/{1}/{2}'.format(
            url,
            urllib.parse.quote(bucket, safe=''),
            quoted
        ),
        headers=headers(key),
        timeout=(10, 120),
    )
    if response.status_code != 200:
        raise RuntimeError(
            'Failed to download {0} ({1}): {2}'.format(
                filename,
                response.status_code,
                response.text
            )
        )
    return response.content
