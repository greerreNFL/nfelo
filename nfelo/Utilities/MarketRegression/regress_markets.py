from .regress_open import regress_open
from .regress_close import regress_close


def _is_missing(value) -> bool:
    '''
    True if value is None or NaN.
    '''
    return value is None or value != value


def regress_markets(
    base_dif: float,
    market_dif_open: float,
    market_dif_close: float,
    mid: float,
    steep: float,
    min_mr: float,
    cap: float,
    close_exp: float,
) -> tuple:
    '''
    Run open then close market regression for one game.

    Open uses the logistic S-curve. Close uses CPR from the open effective
    factor and open-to-close movement. If the open market is missing but
    close exists, close is regressed with an open-style pass against the
    close market (equivalent to zero open-to-close movement). Missing legs
    return None.

    Parameters:
    * base_dif (float): model's raw elo dif before market regression
    * market_dif_open (float): opening market elo dif (blended target)
    * market_dif_close (float): closing market elo dif (blended target)
    * mid (float): logistic midpoint where open factor equals 0.5 (mr_mid)
    * steep (float): logistic steepness in 1/elo units (mr_steep)
    * min_mr (float): logistic floor
    * cap (float): max absolute residual vs market in elo (mr_cap_elo)
    * close_exp (float): CPR movement sensitivity (mr_close_exp)

    Returns:
    * nfelo_dif_open (float or None)
    * market_regression_factor_open (float or None)
    * nfelo_dif_close (float or None)
    * market_regression_factor_close (float or None)
    '''
    has_open = not _is_missing(market_dif_open)
    has_close = not _is_missing(market_dif_close)
    ## open leg ##
    if has_open:
        dif_open, factor_open = regress_open(
            base_dif, market_dif_open, mid, steep, min_mr, cap,
        )
        open_od = abs(base_dif - market_dif_open)
    else:
        dif_open = None
        factor_open = None
        open_od = None
    ## close leg ##
    if has_close and has_open:
        ## CPR from open PR and movement ##
        dif_close, factor_close = regress_close(
            base_dif, market_dif_close,
            factor_open, open_od, close_exp, cap,
        )
    elif has_close and not has_open:
        ## no opener: open-style pass on close (zero-movement equivalent) ##
        dif_close, factor_close = regress_open(
            base_dif, market_dif_close, mid, steep, min_mr, cap,
        )
    else:
        dif_close = None
        factor_close = None
    ## return ##
    return dif_open, factor_open, dif_close, factor_close
