import pandas as pd
from typing import Callable

def merge_check(
    merge_func:Callable[[pd.DataFrame], pd.DataFrame],
    df:pd.DataFrame, 
    process_name:str
):
    '''
    Utility function that prints information status about the merge
    and flags if additional records (ie dupes) were created

    Parameters:
    * merge_func (function): the function that merges new data on the df
    * df (DataFrame): the df to merge to
    * process_name (str): the name of the process for printing

    Returns:
    * df (DataFrame): the original df with the merged data 
    '''
    print('     Merging {0}'.format(process_name))
    pre = len(df)
    df = merge_func(df)
    post = len(df)
    if post-pre > 0:
        print('          Warning: added {0} new records'.format(post-pre))
    ## return ##
    return df