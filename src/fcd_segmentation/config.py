"""Configuration loading and portable path helpers."""
from pathlib import Path
import yaml

def load_yaml(path):
    with Path(path).open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value

def resolve_path(value, base_dir=None):
    path = Path(value).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    return path.resolve()
