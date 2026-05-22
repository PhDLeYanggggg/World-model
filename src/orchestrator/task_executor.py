from __future__ import annotations

import subprocess
import shlex
import time
from pathlib import Path
from typing import Dict

from .task_queue import QueueTask


LOG_DIR = Path("outputs/logs/overnight_stage13")


def execute_task(task: QueueTask, timeout_s: int | None = None) -> Dict:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()
    log_path = LOG_DIR / f"{int(start)}_{task.name}.log"
    result = {
        "task": task.name,
        "priority": task.priority,
        "command": task.command,
        "status": "running",
        "returncode": None,
        "elapsed_seconds": 0.0,
        "log_path": str(log_path),
    }
    try:
        proc = subprocess.run(
            shlex.split(task.command),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
        )
        log_path.write_text(proc.stdout, encoding="utf-8")
        result["returncode"] = proc.returncode
        result["status"] = "completed" if proc.returncode == 0 else "failed"
    except subprocess.TimeoutExpired as exc:
        log_path.write_text((exc.stdout or "") + "\nTIMEOUT\n", encoding="utf-8")
        result["returncode"] = -9
        result["status"] = "timeout"
    except Exception as exc:
        log_path.write_text(str(exc), encoding="utf-8")
        result["returncode"] = -1
        result["status"] = "failed"
        result["error"] = str(exc)
    result["elapsed_seconds"] = round(time.time() - start, 3)
    return result
