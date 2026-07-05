import pathlib
import pandas as pd
from typing import Dict, Optional
from ..constants import CONFIG_DIR
from ..types import PriorConfig
from .helpers.compute_normalization import compute_normalization
from .helpers.load_panel import load_prior_panel
from .helpers.write_config import load_prior_config, save_prior_config


def update_prior(
    config_path: pathlib.Path,
    panel: pd.DataFrame,
) -> dict:
    '''
    Recompute and persist normalization for one prior config file.

    Parameters:
    * config_path (pathlib.Path): path to prior config json
    * panel (pd.DataFrame): team-season panel with raw prior value columns

    Returns:
    * result (dict): normalization table and metadata
    '''
    config = load_prior_config(config_path)
    normalization = compute_normalization(
        panel=panel,
        value_col=config.raw_value_col,
        mu=config.mu,
        half_life=config.training.half_life,
        min_history_seasons=config.training.min_history_seasons,
    )
    trained_through_season = int(panel.dropna(subset=[config.raw_value_col])['season'].max())
    updated = config.with_normalization(normalization, trained_through_season)
    save_prior_config(config_path, updated)
    return {
        'prior': config.prior,
        'normalization': normalization,
        'trained_through_season': trained_through_season,
    }


def update_all_priors(
    panel: Optional[pd.DataFrame] = None,
    config_dir: Optional[pathlib.Path] = None,
) -> Dict[str, dict]:
    '''
    Recompute and persist normalization for all prior config files.

    Parameters:
    * panel (pd.DataFrame): optional team-season panel; loaded if omitted
    * config_dir (pathlib.Path): optional override for config directory

    Returns:
    * results (dict): prior name -> update result
    '''
    config_dir = config_dir or CONFIG_DIR
    panel = panel if panel is not None else load_prior_panel()
    results = {}
    for config_path in sorted(config_dir.glob('*.json')):
        result = update_prior(config_path, panel)
        results[result['prior']] = result
    return results
