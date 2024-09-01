import pandas as pd
import numpy

def grade_se_vector(
      model_line:pd.Series,
      result:pd.Series,
) -> pd.Series:
    '''
    Grades a series of model predictions and results to determine the
    squared error

    Parameters:
    * model_line: The model's predicted spread
    * result: game result

    Returns:
    se: squared error
    '''
    ## get the error ##
    error = model_line + result
    ## return squared error ##
    return error ** 2