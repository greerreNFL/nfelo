import datetime
import json
import sys
import time
from typing import Any, Dict, List

from .Environments import environment_from_config
from .Environments.Job import Job
from .Primitives.RunPlan import RunPlan
from .Primitives.ShardMerger import ShardMerger


class NfeloTraining():
    '''
    Orchestrates a parallel training run: queue shards, respect max_workers,
    poll the environment, merge finished shard CSVs to the run root.
    '''

    def __init__(self, plan: RunPlan):
        self.plan = plan
        env_cfg = dict(plan.environment)
        if plan.max_seconds_per_shard is not None:
            env_cfg.setdefault('max_seconds_per_shard', plan.max_seconds_per_shard)
        self.environment = environment_from_config(env_cfg, plan.repo_root)
        self.max_workers = plan.environment.get('max_workers', 1)

    def run(self) -> Dict[str, Any]:
        ## write plan and build shard queue ##
        self.plan.write()
        queue = [
            self.plan.shard_config(i)
            for i in range(1, self.plan.n_shards + 1)
        ]
        active: List[Job] = []
        finished: List[Job] = []
        failed: List[Dict[str, Any]] = []
        ## submit and poll until every shard finishes ##
        while queue or active:
            while queue and len(active) < self.max_workers:
                shard_config = queue.pop(0)
                job = self.environment.submit(shard_config)
                active.append(job)
            still_active = []
            for job in active:
                status = self.environment.poll(job)
                if status == 'running':
                    still_active.append(job)
                    continue
                if status == 'done':
                    self.environment.collect(job)
                    finished.append(job)
                else:
                    failed.append({
                        'shard_id': job.shard_id,
                        'status': status,
                        'error': job.error,
                        'output_dir': job.output_dir,
                    })
            active = still_active
            if active:
                time.sleep(0.2)
        ## merge finished shard CSVs to run root ##
        merge_error = None
        if finished:
            try:
                ShardMerger.merge(
                    self.plan.run_dir,
                    self.plan.opti_tag,
                    self.plan.opti_date,
                )
            except Exception as exc:
                merge_error = str(exc)
        ## write summary ##
        summary = {
            'run_id': self.plan.run_id,
            'environment': self.plan.environment['type'],
            'submitted': self.plan.n_shards,
            'finished': len(finished),
            'failed': len(failed),
            'run_dir': str(self.plan.run_dir),
            'merge_error': merge_error,
            'failed_shards': failed,
            'completed_at': datetime.datetime.now().isoformat(),
        }
        with open(self.plan.details_dir / 'summary.json', 'w') as fp:
            json.dump(summary, fp, indent=2)
        return summary


def run_training(plan_path: str) -> Dict[str, Any]:
    '''
    Run a parallel training job from a run_details/plan.json path.
    '''
    plan = RunPlan.from_json_file(plan_path)
    return NfeloTraining(plan).run()
