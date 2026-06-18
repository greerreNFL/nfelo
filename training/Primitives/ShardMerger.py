import pathlib
from typing import List, Optional, Union

import pandas as pd


class ShardMerger():
    '''
    Concatenate per-shard optimizer CSVs into the run directory root.
    '''

    _SKIP_SUFFIXES = ('_test', '_benchmarks', '_runtime')

    @classmethod
    def merge(cls, run_dir: pathlib.Path, opti_tag: str, opti_date: str) -> pathlib.Path:
        ## locate shard dirs ##
        details_dir = run_dir / 'run_details'
        shards_dir = details_dir / 'shards'
        shard_dirs = sorted([p for p in shards_dir.glob('shard_*') if p.is_dir()])
        if not shard_dirs:
            raise FileNotFoundError('No shard directories under {0}'.format(shards_dir))
        ## merge train + side tables to run root ##
        stem = '{0}-{1}'.format(opti_tag, opti_date)
        cls._merge_train(shard_dirs, opti_tag, run_dir / '{0}.csv'.format(stem))
        cls._merge_suffix(shard_dirs, opti_tag, '_test', run_dir / '{0}_test.csv'.format(stem), dedupe='run_id')
        cls._merge_suffix(
            shard_dirs, opti_tag, '_benchmarks',
            run_dir / '{0}_benchmarks.csv'.format(stem),
            dedupe=['split', 'model_name'],
        )
        cls._merge_suffix(shard_dirs, opti_tag, '_runtime', run_dir / '{0}_runtime.csv'.format(stem), dedupe=None)
        ## write manifest ##
        cls._write_manifest(details_dir, shard_dirs, opti_tag)
        return run_dir

    @classmethod
    def _merge_train(cls, shard_dirs: List[pathlib.Path], opti_tag: str, out_path: pathlib.Path) -> None:
        frames = []
        for shard_dir in shard_dirs:
            for path in shard_dir.glob('{0}-*.csv'.format(opti_tag)):
                if any(s in path.stem for s in cls._SKIP_SUFFIXES):
                    continue
                df = pd.read_csv(path, index_col=0)
                df['shard'] = shard_dir.name
                frames.append(df)
        if not frames:
            raise FileNotFoundError('No train CSVs found for {0}'.format(opti_tag))
        merged = pd.concat(frames, ignore_index=True)
        if 'run_id' in merged.columns:
            merged = merged.drop_duplicates(subset=['run_id'], keep='last')
        merged.to_csv(out_path)

    @classmethod
    def _merge_suffix(
        cls,
        shard_dirs: List[pathlib.Path],
        opti_tag: str,
        suffix: str,
        out_path: pathlib.Path,
        dedupe: Optional[Union[str, List[str]]],
    ) -> None:
        frames = []
        for shard_dir in shard_dirs:
            matches = list(shard_dir.glob('{0}-*{1}.csv'.format(opti_tag, suffix)))
            if not matches:
                continue
            df = pd.read_csv(matches[0], index_col=0)
            df['shard'] = shard_dir.name
            frames.append(df)
        if not frames:
            return
        merged = pd.concat(frames, ignore_index=True)
        if dedupe:
            keys = dedupe if isinstance(dedupe, list) else [dedupe]
            if all(k in merged.columns for k in keys):
                merged = merged.drop_duplicates(subset=keys, keep='last')
        merged.to_csv(out_path)

    @classmethod
    def _write_manifest(cls, details_dir: pathlib.Path, shard_dirs: List[pathlib.Path], opti_tag: str) -> None:
        lines = ['shard,has_train,has_test,has_benchmarks,has_runtime,has_meta']
        for shard_dir in shard_dirs:
            lines.append('{0},{1},{2},{3},{4},{5}'.format(
                shard_dir.name,
                cls._has_train(shard_dir, opti_tag),
                bool(list(shard_dir.glob('{0}-*_test.csv'.format(opti_tag)))),
                bool(list(shard_dir.glob('{0}-*_benchmarks.csv'.format(opti_tag)))),
                bool(list(shard_dir.glob('{0}-*_runtime.csv'.format(opti_tag)))),
                (shard_dir / 'shard_meta.json').exists(),
            ))
        (details_dir / 'manifest.csv').write_text('\n'.join(lines) + '\n')

    @classmethod
    def _has_train(cls, shard_dir: pathlib.Path, opti_tag: str) -> bool:
        for path in shard_dir.glob('{0}-*.csv'.format(opti_tag)):
            if not any(s in path.stem for s in cls._SKIP_SUFFIXES):
                return True
        return False
