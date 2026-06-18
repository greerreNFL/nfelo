import dataclasses
import datetime
import json
import pathlib
from typing import Any, Dict, List, Optional

from .ShardConfig import ShardConfig


@dataclasses.dataclass
class RunPlan:
    '''
    Describes a full parallel training run and where its artifacts land.
    '''
    run_id: str
    n_shards: int
    repo_root: str
    output_root: str
    environment: Dict[str, Any]
    opti_tag: str
    features: List[str]
    objective: str
    opti_date: str = dataclasses.field(
        default_factory=lambda: datetime.datetime.now().strftime('%Y-%m-%d')
    )
    test_seasons: Optional[List[int]] = None
    bg_overrides: Dict[str, float] = dataclasses.field(default_factory=dict)
    max_seconds_per_shard: Optional[int] = None
    tol: float = 0.000001
    step: float = 0.00001

    @property
    def run_dir(self) -> pathlib.Path:
        return pathlib.Path(self.output_root) / self.run_id

    @property
    def details_dir(self) -> pathlib.Path:
        return self.run_dir / 'run_details'

    @property
    def shards_dir(self) -> pathlib.Path:
        return self.details_dir / 'shards'

    def shard_config(self, shard_id: int) -> ShardConfig:
        return ShardConfig(
            run_id=self.run_id,
            shard_id=shard_id,
            output_dir=str(self.shards_dir / 'shard_{0:03d}'.format(shard_id)),
            repo_root=self.repo_root,
            opti_tag=self.opti_tag,
            opti_date=self.opti_date,
            features=self.features,
            objective=self.objective,
            test_seasons=self.test_seasons,
            bg_overrides=self.bg_overrides,
            max_seconds=self.max_seconds_per_shard,
            tol=self.tol,
            step=self.step,
        )

    def write(self) -> pathlib.Path:
        self.details_dir.mkdir(parents=True, exist_ok=True)
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        path = self.details_dir / 'plan.json'
        with open(path, 'w') as fp:
            json.dump(dataclasses.asdict(self), fp, indent=2)
        return path

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RunPlan':
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in fields})

    @classmethod
    def from_json_file(cls, path: str) -> 'RunPlan':
        with open(path, 'r') as fp:
            return cls.from_dict(json.load(fp))
