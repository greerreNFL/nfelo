import dataclasses
import json
import pathlib
from typing import Any, Dict, List, Optional


@dataclasses.dataclass
class ShardConfig:
    '''
    Serializable config for one parallel shard worker.
    '''
    run_id: str
    shard_id: int
    output_dir: str
    repo_root: str
    opti_tag: str
    opti_date: str
    features: List[str]
    objective: str
    test_seasons: Optional[List[int]] = None
    bg_overrides: Dict[str, float] = dataclasses.field(default_factory=dict)
    max_seconds: Optional[int] = None
    tol: float = 0.000001
    step: float = 0.00001

    def write(self) -> pathlib.Path:
        out = pathlib.Path(self.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / 'shard.json'
        with open(path, 'w') as fp:
            json.dump(dataclasses.asdict(self), fp, indent=2)
        return path

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ShardConfig':
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in fields})

    @classmethod
    def from_json_file(cls, path: str) -> 'ShardConfig':
        with open(path, 'r') as fp:
            return cls.from_dict(json.load(fp))
