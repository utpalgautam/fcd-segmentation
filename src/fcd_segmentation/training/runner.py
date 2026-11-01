"""Execution runner for nnU-Net commands with logging, environment propagation, and error reporting."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fcd_segmentation.training.paths import NNUNetPaths

logger = logging.getLogger(__name__)


@dataclass
class CommandResult:
    command: list[str]
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    start_time: str
    end_time: str
    env_vars: dict[str, str] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.exit_code == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_seconds": self.duration_seconds,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "env_vars": self.env_vars,
            "success": self.success,
        }


def run_command(
    cmd: list[str],
    paths: NNUNetPaths,
    log_file: Path | None = None,
    extra_env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> CommandResult:
    """Execute a command via subprocess.run with argument list, passing nnU-Net env variables.

    Captures stdout/stderr, logs execution, and writes output to log_file if provided.
    """
    env = os.environ.copy()
    env.update(paths.to_env_dict())
    if extra_env:
        env.update(extra_env)

    # Convert cmd elements to strings
    str_cmd = [str(x) for x in cmd]
    cmd_str_repr = " ".join(str_cmd)

    start_dt = datetime.now(UTC)
    start_time_iso = start_dt.isoformat()
    t0 = time.time()

    logger.info("Executing command: %s", cmd_str_repr)
    logger.debug("Effective environment: %s", paths.to_env_dict())

    try:
        proc = subprocess.run(
            str_cmd,
            capture_output=True,
            text=True,
            env=env,
            cwd=str(cwd or paths.project_root),
            check=False,
        )
        t1 = time.time()
        end_dt = datetime.now(UTC)
        end_time_iso = end_dt.isoformat()
        duration = t1 - t0

        result = CommandResult(
            command=str_cmd,
            exit_code=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            duration_seconds=duration,
            start_time=start_time_iso,
            end_time=end_time_iso,
            env_vars=paths.to_env_dict(),
        )

        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with log_file.open("a", encoding="utf-8") as f:
                f.write(f"\n--- Command: {cmd_str_repr} ---\n")
                f.write(
                    f"Start: {start_time_iso} | End: {end_time_iso} | Duration: {duration:.2f}s | Exit: {proc.returncode}\n"
                )
                f.write("=== STDOUT ===\n")
                f.write(proc.stdout)
                f.write("\n=== STDERR ===\n")
                f.write(proc.stderr)
                f.write("\n")

        if proc.returncode != 0:
            logger.error(
                "Command failed with exit code %d: %s\nStderr: %s",
                proc.returncode,
                cmd_str_repr,
                proc.stderr.strip() or proc.stdout.strip(),
            )
        else:
            logger.info("Command completed successfully in %.2fs", duration)

        return result

    except Exception as e:
        t1 = time.time()
        duration = t1 - t0
        end_time_iso = datetime.now(UTC).isoformat()
        err_msg = f"Exception while invoking command {cmd_str_repr}: {e}"
        logger.error(err_msg, exc_info=True)

        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with log_file.open("a", encoding="utf-8") as f:
                f.write(f"\n--- Exception executing {cmd_str_repr} ---\n{err_msg}\n")

        return CommandResult(
            command=str_cmd,
            exit_code=-1,
            stdout="",
            stderr=err_msg,
            duration_seconds=duration,
            start_time=start_time_iso,
            end_time=end_time_iso,
            env_vars=paths.to_env_dict(),
        )


def find_executable(name: str) -> str:
    """Find executable in active virtual environment or PATH, or fallback to python -m module."""
    venv_bin = Path(sys.executable).parent
    # On Windows, check for .exe
    if sys.platform == "win32":
        candidate = venv_bin / f"{name}.exe"
        if candidate.is_file():
            return str(candidate)
    candidate = venv_bin / name
    if candidate.is_file():
        return str(candidate)
    return name
