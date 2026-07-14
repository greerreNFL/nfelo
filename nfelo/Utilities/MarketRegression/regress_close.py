## disagreement below this is treated as exact agreement ##
EPS = 1e-9


def close_pr(
    pr: float,
    od: float,
    cd: float,
    close_exp: float,
) -> float:
    '''
    Rescale the open effective factor (PR) by the open-to-close movement
    ratio through an odds transform (CPR).

    r = CD / OD. Movement toward the model (r < 1) restores conviction;
    movement away (r > 1) cushions toward the close market.

    Parameters:
    * pr (float): effective open regression factor
    * od (float): absolute opening disagreement |base - market_open|
    * cd (float): absolute closing disagreement |base - market_close|
    * close_exp (float): movement sensitivity (mr_close_exp)

    Returns:
    * cpr (float): close regression factor
    '''
    ## no opening opinion -> movement cannot create one ##
    if od <= EPS:
        return 1.0
    ## fully regressed opener stays fully regressed ##
    if pr >= 1.0:
        return 1.0
    ## movement ratio ##
    r = cd / od
    ## odds transform: CPR = PR * r^exp / (PR * r^exp + (1 - PR)) ##
    weighted = pr * (r ** close_exp)
    return weighted / (weighted + (1.0 - pr))


def regress_close(
    base_dif: float,
    market_dif: float,
    pr: float,
    od: float,
    close_exp: float,
    cap: float,
) -> tuple[float, float]:
    '''
    Regress the model's raw elo dif toward the closing market elo dif
    using CPR (open PR rescaled by movement).

    The close opinion is always anchored between base_dif and market_dif
    (pre-cap). The posted residual vs market is capped at +/- cap elo.

    Parameters:
    * base_dif (float): model's raw elo dif before market regression
    * market_dif (float): closing market elo dif (blended target)
    * pr (float): effective open regression factor from regress_open
    * od (float): absolute opening disagreement used to form PR
    * close_exp (float): movement sensitivity (mr_close_exp)
    * cap (float): max absolute residual vs market in elo (mr_cap_elo)

    Returns:
    * regressed_dif (float): close-regressed elo dif
    * regression_factor (float): effective close factor CPR
    '''
    ## closing disagreement ##
    cd = abs(base_dif - market_dif)
    ## CPR from open PR and movement ##
    cpr = close_pr(pr, od, cd, close_exp)
    ## regress toward the closing market ##
    regressed_dif = base_dif + cpr * (market_dif - base_dif)
    ## cap residual vs market ##
    residual = regressed_dif - market_dif
    if residual > cap:
        residual = cap
    elif residual < -cap:
        residual = -cap
    regressed_dif = market_dif + residual
    ## effective post-cap factor ##
    if cd <= EPS:
        regression_factor = 1.0
    else:
        regression_factor = (regressed_dif - base_dif) / (market_dif - base_dif)
    ## return ##
    return regressed_dif, regression_factor
