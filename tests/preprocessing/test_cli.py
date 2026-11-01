"""Unit tests for the prepare_dataset.py CLI entry point."""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLI_SCRIPT = PROJECT_ROOT / "scripts" / "prepare_dataset.py"


def test_cli_help():
    proc = subprocess.run(
        [sys.executable, str(CLI_SCRIPT), "--help"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "FCD MRI Preprocessing Pipeline" in proc.stdout
    assert "--inventory-only" in proc.stdout
    assert "--limit" in proc.stdout
    assert "--resume" in proc.stdout
    assert "--overwrite" in proc.stdout


def test_cli_fails_on_missing_config(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            str(CLI_SCRIPT),
            "--config",
            str(tmp_path / "nonexistent_config.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "Configuration file not found" in proc.stderr


def test_cli_fails_on_missing_input_dir(tmp_path):
    cfg_path = PROJECT_ROOT / "configs" / "preprocessing_config.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(CLI_SCRIPT),
            "--config",
            str(cfg_path),
            "--input",
            str(tmp_path / "nonexistent_raw_dir"),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "Input cohort directory not found" in proc.stderr
