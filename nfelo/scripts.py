import pathlib

from .storage import (
    OUTPUT_FILES,
    download_output_file,
    output_folder,
)


def fetch_nfelo():
    '''
    Download the current nfelo output CSVs from storage into output_data/.

    Requires STORAGE_URL, STORAGE_KEY, and NFELO_STORAGE_PATH.
    '''
    folder = output_folder()
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
    for filename in OUTPUT_FILES:
        content = download_output_file(filename)
        dest = '{0}/{1}'.format(folder, filename)
        with open(dest, 'wb') as f:
            f.write(content)
    return folder
