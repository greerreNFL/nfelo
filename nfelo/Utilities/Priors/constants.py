import pathlib

PRIORS_DIR = pathlib.Path(__file__).parent.resolve()
CONFIG_DIR = PRIORS_DIR / 'config'
DEFAULT_MIN_HISTORY_SEASONS = 2
DEFAULT_ELO_CENTER = 1505
