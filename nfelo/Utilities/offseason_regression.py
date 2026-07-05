import pandas as pd
from dataclasses import dataclass
from typing import List
from .Priors import prior_to_elo


@dataclass
class ExternalPrior:
    '''
    Input format for external priors in the offseason regression function
    '''
    prior_type: str ## the type of prior
    weight: float ## the weight of the prior
    value: float ## the value of the prior

@dataclass
class ResolvedPrior:
    '''
    An external prior after normalization to elo (and any weight
    renormalization) inside the offseason regression function
    '''
    prior_type: str ## the type of prior
    weight: float ## the weight of the prior, renormalized if weights summed over 1
    value: float ## the raw value of the prior
    elo: float ## the prior normalized to elo

@dataclass
class OffseasonReversionResult:
    '''
    A return object to pass information about the reversion back
    to the nfelo model for state tracking and analytics
    '''
    new_elo: float ## what the team's elo was updated to
    league_elo: float ## the league median elo for the season
    previous_elo: float ## the team's elo from the previous season
    mean_reverted_elo: float ## the team's elo after internal mean reversion
    external_priors: List[ResolvedPrior] ## the external priors that were used to update the team's elo
    reverted_weight: float ## the weight of the internal mean reversion

    def to_record(self, team, season, week):
        '''
        Converts the reversion result to a record for state tracking and analytics
        '''
        record = {
            'team': team,
            'season': season,
            'week': week,
            'previous_ending_elo': self.previous_elo,
            'league_elo': self.league_elo,
            'mean_reverted_elo': self.mean_reverted_elo,
            'new_elo': self.new_elo,
        }
        for prior in self.external_priors:
            record['{0}_elo'.format(prior.prior_type)] = prior.elo
        return record


def offseason_regression(
    league_elo: float,
    previous_elo: float,
    reversion: float,
    season: int,
    prior_elo_scale: float,
    external_priors: List[ExternalPrior],
) -> OffseasonReversionResult:
    '''
    Regresses a team's elo from one season to the next by creating a weighted
    average of mean reverted previous elo and normalized external priors.

    Parameters:
    * league_elo (float): the previous year's league median elo
    * previous_elo (float): the team ending elo from last year
    * reversion (float): season over season mean reversion
    * season (int): season for prior normalization lookup
    * prior_elo_scale (float): z -> elo multiplier from nfelo config
    * external_priors (List[ExternalPrior]): the external priors to blend

    Returns:
    * OffseasonReversionResult
    '''
    ## normalize the previous year elo to get the model back to 1505 median
    ## note, all calculations from an elo model are denominated in elo differences, so
    ## rebasing to a 1505 median has no effect on predictions, it is simply for
    ## year to year consistency
    previous_elo_norm = 1505 + (previous_elo - league_elo)
    ## apply the internal mean reversion to create the base prior for next year ##
    mean_reverted_elo = (
        reversion * 1505 +
        (1 - reversion) * previous_elo_norm
    )
    ## normalize external priors to elo ##
    ## normalization translates a z-score into elo scale
    resolved = []
    for prior in external_priors:
        elo = prior_to_elo(
            season,
            prior.value,
            prior.prior_type,
            prior_elo_scale,
            mean_reverted_elo,
        )
        ## add resolved prior to list ##
        resolved.append(ResolvedPrior(
            prior_type=prior.prior_type,
            weight=prior.weight,
            value=prior.value,
            elo=elo,
        ))
    ## normalize the weights if over 1 ##
    total_config_weight = sum(p.weight for p in resolved)
    if total_config_weight > 1:
        for prior in resolved:
            prior.weight = prior.weight / total_config_weight
        total_config_weight = 1
    ## calc implied reversion weight ##
    reverted_weight = 1 - total_config_weight
    ## calc the weighted avg ##
    new_elo = reverted_weight * mean_reverted_elo
    for prior in resolved:
        new_elo += prior.weight * prior.elo
    ## return ##
    return OffseasonReversionResult(
        new_elo=new_elo,
        league_elo=league_elo,
        previous_elo=previous_elo,
        mean_reverted_elo=mean_reverted_elo,
        external_priors=resolved,
        reverted_weight=reverted_weight,
    )
