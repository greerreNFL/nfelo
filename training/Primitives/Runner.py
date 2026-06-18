import json
import pathlib
import sys
from typing import Any, Dict

import numpy

from .ShardConfig import ShardConfig


class Runner():
    '''
    Executes one random-start SLSQP hop and writes optimizer output to the
    shard directory. hop_number is set to shard_id so run_id values stay
    unique across parallel shards (see TRAINING_PLAYBOOK.md).
    '''

    @classmethod
    def run(cls, shard_config: ShardConfig) -> Dict[str, Any]:
        ## load config ##
        repo_root = pathlib.Path(shard_config.repo_root)
        config_loc = repo_root / 'config.json'
        with open(config_loc, 'r') as fp:
            config = json.load(fp)
        ## load data ##
        from nfelo.Data import DataLoader
        from nfelo.Model import Nfelo
        from nfelo.Optimizer import NfeloOptimizer
        data = DataLoader()
        nfelo = Nfelo(data=data, config=config['models']['nfelo'])
        ## optimize ##
        optimizer = NfeloOptimizer(
            shard_config.opti_tag,
            nfelo,
            shard_config.features,
            shard_config.objective,
            bg_overrides=shard_config.bg_overrides,
            test_seasons=shard_config.test_seasons,
        )
        base = optimizer.base
        base.results_dir = pathlib.Path(shard_config.output_dir)
        base.hop_number = shard_config.shard_id
        base.tol = shard_config.tol
        base.step = shard_config.step
        base.opti_date = shard_config.opti_date
        ## one random-start hop per shard; parallel shards replace RandomStarts' serial loop ##
        ## draw here so hop_number can stay shard_id (RandomStarts overwrites hop_number) ##
        rng = numpy.random.default_rng()
        base.best_guesses = rng.uniform(
            0.0, 1.0, size=len(shard_config.features)
        ).tolist()
        optimizer.optimize()
        ## write shard meta ##
        meta = {
            'run_id': shard_config.run_id,
            'shard_id': shard_config.shard_id,
            'opti_seconds': base.opti_seconds,
            'status': 'ok',
        }
        out = pathlib.Path(shard_config.output_dir)
        with open(out / 'shard_meta.json', 'w') as fp:
            json.dump(meta, fp, indent=2)
        return meta


def run_shard(config_path: str) -> Dict[str, Any]:
    '''
    Run one shard worker from a shard.json path.
    '''
    return Runner.run(ShardConfig.from_json_file(config_path))


if __name__ == '__main__':
    run_shard(sys.argv[1])
