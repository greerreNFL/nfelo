import pathlib
import pandas as pd
import nfelodcm as dcm


def load_prior_panel() -> pd.DataFrame:
    '''
    Team-season panel of raw prior values for normalization updates.

    Returns:
    * panel (pd.DataFrame): team, season, and raw prior value columns
    '''
    nfelo_package_dir = pathlib.Path(__file__).parent.parent.parent.parent.parent.resolve()
    intermediate = nfelo_package_dir / 'Data' / 'Intermediate Data'
    db = dcm.load(['wt_ratings', 'nfelounits_elo'])
    ## load wt ratings ##
    wt = db['wt_ratings'][['team', 'season', 'wt_rating']].copy()
    ## load dvoa projections ##
    dvoa = pd.read_csv(intermediate / 'dvoa_projections.csv', index_col=0)
    dvoa = dvoa[['team', 'season', 'projected_total_dvoa']].copy()
    dvoa = dvoa.rename(columns={'projected_total_dvoa': 'projected_dvoa'})
    ## load units preseason elo ##
    units_pre = db['nfelounits_elo'].sort_values(['team', 'season', 'week']).groupby(
        ['team', 'season'], as_index=False
    ).first()[['team', 'season', 'elo']].rename(columns={'elo': 'units_preseason_elo'})
    ## merge into one panel ##
    panel = wt.merge(dvoa, on=['team', 'season'], how='outer')
    panel = panel.merge(units_pre, on=['team', 'season'], how='outer')
    return panel.sort_values(['season', 'team']).reset_index(drop=True)
