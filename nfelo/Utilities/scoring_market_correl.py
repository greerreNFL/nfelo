import pandas as pd
import numpy
from .base import is_series

def market_correl(
      model_line:pd.Series,
      market_line:pd.Series,
) -> pd.Series:
    '''
    Calculates the correlation between the model and the market

    Parameters:
    * model_line: model's home spread
    * market_line: market's home spread

    Returns:
    * correl: model<>market correlation
    '''
    return model_line.corr(market_line)
