import pandas as pd
import numpy

def grade_su_vector(
      model_line:pd.Series,
      result:pd.Series,
) -> pd.Series:
    '''
    Grades a series of model predictions and results to determine how many
    games the model correctly picked straight up

    Parameters:
    * model_line: The model's predicted spread
    * result: game result

    Returns:
    su: a series of 1,0, and nans representing correct, wrong, and push
    '''
    ## flip model line to represent a result instead of spread ##
    expected_result = model_line * -1
    ## formulate logic ##
    return numpy.where(
        expected_result == result,
        numpy.nan,
        numpy.where(
            ## if the expected and actual are both positive, thats
            ## correct home win. If both negative, a correctly picked home
            ## loss
            numpy.sign(expected_result) == numpy.sign(result),
            1,
            0
        )
    )