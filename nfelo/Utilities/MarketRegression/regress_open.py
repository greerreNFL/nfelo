from .s_curve import s_curve_factor

## disagreement below this is treated as exact agreement ##
EPS = 1e-9


def regress_open(
    base_dif: float,
    market_dif: float,
    mid: float,
    steep: float,
    min_mr: float,
    cap: float,
) -> tuple[float, float]:
    '''
    Regress the model's raw elo dif toward the opening market elo dif
    using the logistic S-curve factor.

    The posted residual vs market is capped at +/- cap elo. The returned
    regression_factor is the effective post-cap factor (PR), which the
    close leg consumes.

    Parameters:
    * base_dif (float): model's raw elo dif before market regression
    * market_dif (float): opening market elo dif (blended target)
    * mid (float): logistic midpoint where factor equals 0.5 (mr_mid)
    * steep (float): logistic steepness in 1/elo units (mr_steep)
    * min_mr (float): logistic floor
    * cap (float): max absolute residual vs market in elo (mr_cap_elo)

    Returns:
    * regressed_dif (float): open-regressed elo dif
    * regression_factor (float): effective open factor PR in [min_mr, 1]
    '''
    ## opening disagreement ##
    od = abs(base_dif - market_dif)
    ## logistic S-curve factor from disagreement ##
    factor = s_curve_factor(od, mid, steep, min_mr)
    ## regress toward the opening market ##
    regressed_dif = base_dif + factor * (market_dif - base_dif)
    ## cap residual vs market ##
    residual = regressed_dif - market_dif
    if residual > cap:
        residual = cap
    elif residual < -cap:
        residual = -cap
    regressed_dif = market_dif + residual
    ## effective post-cap factor (PR) ##
    if od <= EPS:
        regression_factor = 1.0
    else:
        regression_factor = (regressed_dif - base_dif) / (market_dif - base_dif)
    ## return ##
    return regressed_dif, regression_factor
