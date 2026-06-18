import datetime
import os
import pathlib
import signal
import subprocess
from typing import Any, Dict, Optional

from ..Primitives.ShardConfig import ShardConfig
from ..Primitives.WorkerCommand import build_worker_argv
from .Base import Environment
from .Job import Job


class LocalEnvironment(Environment):
    '''
    Launch shard workers as host subprocesses in the repo conda env.
    '''

    def __init__(self, config: Dict[str, Any], repo_root: str):
        super().__init__(config, repo_root)
        self.worker_env = dict(config.get('worker_env', {}))
        self.worker_env.setdefault('OMP_NUM_THREADS', '1')

    def submit(self, shard_config: ShardConfig) -> Job:
        ## build worker command ##
        argv = build_worker_argv(shard_config)
        ## env: repo on PYTHONPATH, worker env overrides ##
        env = os.environ.copy()
        env.update(self.worker_env)
        env['PYTHONPATH'] = self.repo_root + (
            os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else ''
        )
        ## launch subprocess, logs to shard dir ##
        out_dir = pathlib.Path(shard_config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stdout_fp = open(out_dir / 'stdout.log', 'w')
        stderr_fp = open(out_dir / 'stderr.log', 'w')
        proc = subprocess.Popen(
            argv,
            cwd=self.repo_root,
            env=env,
            stdout=stdout_fp,
            stderr=stderr_fp,
            text=True,
        )
        return Job(
            shard_id=shard_config.shard_id,
            config_path=str(shard_config.write()),
            output_dir=shard_config.output_dir,
            status='running',
            handle=proc,
            started_at=datetime.datetime.now(),
            stdout_fp=stdout_fp,
            stderr_fp=stderr_fp,
        )

    def poll(self, job: Job) -> str:
        proc = job.handle
        ## still running: check timeout ##
        if proc.poll() is None:
            if self._timed_out(job):
                self.terminate(job)
                job.close_logs()
                job.status = 'timeout'
                job.error = self._read_stderr(job)
                return job.status
            job.status = 'running'
            return job.status
        ## process exited ##
        job.exit_code = proc.returncode
        job.finished_at = datetime.datetime.now()
        job.close_logs()
        if proc.returncode == 0:
            job.status = 'done'
        else:
            job.error = self._read_stderr(job) or 'non-zero exit'
            job.status = 'failed'
        return job.status

    def collect(self, job: Job) -> str:
        return job.output_dir

    def terminate(self, job: Job) -> None:
        proc = job.handle
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        job.close_logs()

    def _read_stderr(self, job: Job) -> Optional[str]:
        err_path = pathlib.Path(job.output_dir) / 'stderr.log'
        if not err_path.exists():
            return None
        text = err_path.read_text().strip()
        return text or None

    def _timed_out(self, job: Job) -> bool:
        if not self.max_seconds or not job.started_at:
            return False
        elapsed = (datetime.datetime.now() - job.started_at).total_seconds()
        return elapsed > self.max_seconds
