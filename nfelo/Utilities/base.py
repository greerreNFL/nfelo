import pandas as pd
import numpy


def is_series(arg):
    '''
    Determines if the passed input is a number or a series

    Parameter:
    * arg (int, float, series, etc): input in question

    Output:
    * is_series (bool): true/false describing whether or not the input is a series
    '''
    ## data types considered numbers ##
    number_types = (
        ## ints ##
        int, numpy.int8, numpy.int16, numpy.int32, numpy.int64,
        ## floats ##
        float, numpy.float16, numpy.float32, numpy.float64
    )
    ## data types considered series-like ##
    series_types = (
        list, pd.Series, numpy.array
    )
    if isinstance(arg, number_types):
        return False
    elif isinstance(arg, series_types):
        return True
    else:
        raise Exception('INPUT ERROR: Passed input ({0}) is of type {1}, but number or series-like is required'.format(
            arg, type(arg)
        ))
        