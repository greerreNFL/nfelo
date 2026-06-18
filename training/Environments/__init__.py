from typing import Any, Dict, Type

from .Base import Environment
from .Local import LocalEnvironment

_ENVIRONMENTS: Dict[str, Type[Environment]] = {
    'local': LocalEnvironment,
}


def environment_from_config(environment_cfg: Dict[str, Any], repo_root: str) -> Environment:
    env_type = environment_cfg['type']
    if env_type not in _ENVIRONMENTS:
        raise ValueError('Unknown environment type: {0}'.format(env_type))
    return _ENVIRONMENTS[env_type](environment_cfg, repo_root)
