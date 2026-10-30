import pytest

from fcd_segmentation.runtime import get_device


def test_invalid_device():
    with pytest.raises(ValueError):
        get_device("tpu")
