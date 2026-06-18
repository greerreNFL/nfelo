import sys
from typing import List

from .ShardConfig import ShardConfig


def build_worker_argv(shard_config: ShardConfig) -> List[str]:
    '''
    Same command shape for every environment. Only the launcher differs.
    '''
    config_path = shard_config.write()
    return [
        sys.executable, '-m', 'training.Primitives.Runner',
        str(config_path),
    ]
