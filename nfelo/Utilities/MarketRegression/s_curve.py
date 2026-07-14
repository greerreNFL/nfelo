import math


def s_curve_factor(
    abs_dif: float,
    mid: float,
    steep: float,
    min_mr: float,
) -> float:
    '''
    Logistic S-curve market regression factor as a function of absolute
    elo disagreement.

    Asymptotes are 1.0 (full regression) and min_mr. The logit center is
    shifted so the factor equals 0.5 exactly at abs_dif = mid.

    Parameters:
    * abs_dif (float): absolute elo disagreement between model and market
    * mid (float): disagreement at which the factor equals 0.5
    * steep (float): logistic steepness in 1/elo units (mr_steep)
    * min_mr (float): asymptotic floor for the regression factor

    Returns:
    * factor (float): regression factor in (min_mr, 1]
    '''
    ## ceiling / floor of the logistic ##
    hi = 1.0
    lo = min_mr
    ## unit sigmoid value needed at mid: (0.5 - lo) / (hi - lo) ##
    s_target = (0.5 - lo) / (hi - lo)
    ## 1 / (1 + exp(steep*(mid-x0))) = s_target  ->  solve for x0 ##
    ratio = (1.0 / s_target) - 1.0
    x0 = mid - math.log(ratio) / steep
    ## standard logistic — clamp exponent to avoid overflow at extreme params ##
    z = steep * (abs_dif - x0)
    if z > 700:
        unit = 0.0
    elif z < -700:
        unit = 1.0
    else:
        unit = 1.0 / (1.0 + math.exp(z))
    return lo + (hi - lo) * unit
