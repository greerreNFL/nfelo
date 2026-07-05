import json
import pathlib
from dataclasses import dataclass, replace
from datetime import date
from typing import Dict, Optional
from .constants import DEFAULT_MIN_HISTORY_SEASONS


@dataclass
class PriorMeta:
    '''
    Metadata for when a prior config normalization table was last updated.
    '''
    trained_through_season: int ## last season included in update panel
    trained_at: str ## iso date of last update


@dataclass
class PriorTraining:
    '''
    Settings used when recomputing normalization tables.
    '''
    half_life: Optional[float] ## decay for weighting past seasons
    min_history_seasons: int ## minimum prior seasons required before emitting sigma


@dataclass
class PriorConfig:
    '''
    Full prior config payload stored in Utilities/Priors/config.
    '''
    prior: str ## prior key used at runtime
    raw_value_col: str ## panel column for raw prior values
    mu: float ## fixed prior mean
    meta: PriorMeta
    training: PriorTraining
    normalization: Dict[str, float] ## season -> sigma

    @classmethod
    def from_path(cls, config_path: pathlib.Path) -> 'PriorConfig':
        with open(config_path, 'r') as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_dict(cls, data: dict) -> 'PriorConfig':
        meta = data.get('meta', {})
        training = data.get('training', {})
        return cls(
            prior=data['prior'],
            raw_value_col=data['raw_value_col'],
            mu=float(data['mu']),
            meta=PriorMeta(
                trained_through_season=int(meta.get('trained_through_season', 0)),
                trained_at=str(meta.get('trained_at', '')),
            ),
            training=PriorTraining(
                half_life=training.get('half_life'),
                min_history_seasons=int(training.get('min_history_seasons', DEFAULT_MIN_HISTORY_SEASONS)),
            ),
            normalization={
                str(season): float(sigma)
                for season, sigma in data.get('normalization', {}).items()
            },
        )

    def to_dict(self) -> dict:
        ## preserve config field order on write ##
        return {
            'prior': self.prior,
            'raw_value_col': self.raw_value_col,
            'mu': self.mu,
            'meta': {
                'trained_through_season': int(self.meta.trained_through_season),
                'trained_at': self.meta.trained_at,
            },
            'training': {
                'half_life': self.training.half_life,
                'min_history_seasons': int(self.training.min_history_seasons),
            },
            'normalization': self.normalization,
        }

    def with_half_life(self, half_life: Optional[float]) -> 'PriorConfig':
        return replace(
            self,
            training=replace(self.training, half_life=half_life),
        )

    def with_normalization(
        self,
        normalization: Dict[int, float],
        trained_through_season: int,
    ) -> 'PriorConfig':
        return replace(
            self,
            meta=PriorMeta(
                trained_through_season=int(trained_through_season),
                trained_at=date.today().isoformat(),
            ),
            normalization={
                str(season): round(float(sigma), 8)
                for season, sigma in sorted(normalization.items())
            },
        )
