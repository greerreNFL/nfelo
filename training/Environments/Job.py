import dataclasses
import datetime
from typing import Any, Optional, TextIO


@dataclasses.dataclass
class Job:
    '''
    Tracks one submitted shard while the environment runs it.
    '''
    shard_id: int
    config_path: str
    output_dir: str
    status: str = 'pending'
    handle: Any = None
    exit_code: Optional[int] = None
    error: Optional[str] = None
    started_at: Optional[datetime.datetime] = None
    finished_at: Optional[datetime.datetime] = None
    stdout_fp: Optional[TextIO] = None
    stderr_fp: Optional[TextIO] = None

    def close_logs(self) -> None:
        for fp in (self.stdout_fp, self.stderr_fp):
            if fp is not None:
                fp.close()
