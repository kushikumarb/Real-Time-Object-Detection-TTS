import pytest
from src.utils import calculate_fps, validate_model
import os


def test_calculate_fps():
    assert 10.0 == pytest.approx(calculate_fps(0.0, 0.1), 0.1)


def test_validate_model(tmp_path):
    fake_model = tmp_path / "fake.pt"
    assert not validate_model(str(fake_model))
