from abc import ABC, abstractmethod
from typing import Any, Dict

from ..Primitives.ShardConfig import ShardConfig
from .Job import Job


class Environment(ABC):
    '''
    How a single ShardConfig gets launched and its artifacts returned.
    '''

    def __init__(self, config: Dict[str, Any], repo_root: str):
        self.config = config
        self.repo_root = repo_root
        self.max_seconds = config.get('max_seconds_per_shard')

    @abstractmethod
    def submit(self, shard_config: ShardConfig) -> Job:
        ...

    @abstractmethod
    def poll(self, job: Job) -> str:
        ...

    @abstractmethod
    def collect(self, job: Job) -> str:
        ...

    @abstractmethod
    def terminate(self, job: Job) -> None:
        ...
