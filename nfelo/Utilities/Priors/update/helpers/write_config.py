import json
import pathlib
from ...types import PriorConfig


def load_prior_config(config_path: pathlib.Path) -> PriorConfig:
    return PriorConfig.from_path(config_path)


def save_prior_config(config_path: pathlib.Path, config: PriorConfig) -> pathlib.Path:
    '''
    Persist a prior config dataclass to disk.

    Parameters:
    * config_path (pathlib.Path): path to prior config json
    * config (PriorConfig): prior config payload

    Returns:
    * path (pathlib.Path): written file path
    '''
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
        f.write('\n')
    return config_path
