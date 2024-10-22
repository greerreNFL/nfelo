import pandas as pd
import numpy
import pathlib

from ...Utilities import (
    elo_to_prob, probability_to_spread,
    calc_shift
)

### Explore how the market resist factor responds to different ###
### scenarios ##
def market_resist_explore():
    recs = []
    ## construct hypotheticals ##
    for home_elo_dif_model in range(-25,26):
        ## scale up to elo dif ##
        home_elo_dif_model *= 20
        for home_elo_dif_market in range(-25,26):
            ## scale up to elo dif ##
            home_elo_dif_market *= 20
            ## transalte to hypothetical spread ##
            try:
                model_spread = probability_to_spread(elo_to_prob(home_elo_dif_model))
                market_spread = probability_to_spread(elo_to_prob(home_elo_dif_market))
                for home_result in range(-10,11):
                    ## calculate hypothetical shifts ##
                    home_shift = calc_shift(
                        home_result*3.0,
                        model_spread,
                        market_spread,
                        9.114,
                        10,
                        2.5,
                        True
                    )
                    ## see what the model would shift without any k adjustment
                    home_shift_model_ex_market = calc_shift(
                        home_result*3.0,
                        model_spread,
                        model_spread,
                        9.114,
                        10,
                        2.5,
                        True
                    )
                    ## see what the market shift would be
                    home_shift_market = calc_shift(
                        home_result*3.0,
                        market_spread,
                        market_spread,
                        9.114,
                        10,
                        2.5,
                        True
                    )
                    ## create the record ##
                    recs.append({
                        'result' : home_result*3.0,
                        'model_elo_dif' : home_elo_dif_model,
                        'model_spread' : model_spread,
                        'model_error_abs' : abs(model_spread+home_result*3),
                        'market_elo_dif' : home_elo_dif_market,
                        'market_spread' : market_spread,
                        'market_abs_error' : abs(market_spread+home_result*3),
                        'home_shift' : home_shift,
                        'home_shift_ex_market' : home_shift_model_ex_market,
                        'home_shift_market' : home_shift_market
                    })
            except:
                print('Could not calc for model_dif: {0} // market_dif: {1}'.format(home_elo_dif_model, home_elo_dif_market))
    ## save ##
    df = pd.DataFrame(recs)
    df['market_resist_impact'] = df['home_shift'] - df['home_shift_ex_market']
    df['new_elo_dif_model'] = df['model_elo_dif'] + df['home_shift'] * 2
    df['new_elo_dif_market'] = df['market_elo_dif'] + df['home_shift_market'] * 2
    df['over_correct'] = numpy.where(
        df['model_elo_dif'] <= df['market_elo_dif'],
        numpy.where(
            df['new_elo_dif_model'] > df['new_elo_dif_market'],
            df['new_elo_dif_model'] - df['new_elo_dif_market'],
            numpy.nan
        ),
        numpy.where(
            df['new_elo_dif_model'] <= df['new_elo_dif_market'],
            df['new_elo_dif_market'] - df['new_elo_dif_model'],
            numpy.nan
        ),
    )
    df.to_csv(
        '{0}/market_resist.csv'.format(
            pathlib.Path(__file__).parent.resolve()
        )
    )

                      
