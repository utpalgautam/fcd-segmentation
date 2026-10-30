from pathlib import Path

import pytest

from fcd_segmentation.config import load_yaml, resolve_path


def test_yaml_mapping(tmp_path: Path):
    config = tmp_path / "config.yaml"
    config.write_text("device: auto\nseed: 42\n", encoding="utf-8")
    assert load_yaml(config) == {"device": "auto", "seed": 42}


def test_yaml_non_mapping(tmp_path: Path):
    config = tmp_path / "config.yaml"
    config.write_text("- item\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_yaml(config)


def test_resolve_path(tmp_path: Path):
    assert resolve_path("data", base_dir=tmp_path) == (tmp_path / "data").resolve()
