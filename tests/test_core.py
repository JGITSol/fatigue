import pytest
from src.config import AppConfig, AlgorithmType
from src.core.detection.analyzer import FatigueAnalyzer

@pytest.fixture
def sample_config():
    return AppConfig(
        model_path="models/test_model.tflite",
        debug_mode=True
    )

def test_ear_calculation(sample_config):
    analyzer = FatigueAnalyzer(sample_config)
    test_eye_open = [(100, 100), (120, 80), (140, 100), (120, 120)]
    ear = analyzer._eye_aspect_ratio(test_eye_open)
    assert 0.2 < ear < 0.4