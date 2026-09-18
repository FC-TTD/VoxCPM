import numpy as np
import pytest

from hub_runtime.timing import apply_timing_control


def tone(seconds=1.0, sample_rate=16000):
    t = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
    return np.sin(2 * np.pi * 440 * t).astype(np.float32)


def test_speed_shortens_audio_and_reports_metadata():
    audio, meta = apply_timing_control(tone(), 16000, speed=1.25)

    assert len(audio) / 16000 == pytest.approx(0.8, abs=0.01)
    assert meta["speed"] == pytest.approx(1.25)
    assert meta["expected_duration"] is None
    assert meta["original_duration_seconds"] == pytest.approx(1.0)
    assert meta["final_duration_seconds"] == pytest.approx(0.8, abs=0.01)
    assert meta["final_speed_factor"] == pytest.approx(1.25)


def test_expected_duration_overrides_speed():
    audio, meta = apply_timing_control(tone(), 16000, speed=1.25, expected_duration=2.0)

    assert len(audio) / 16000 == pytest.approx(2.0, abs=0.02)
    assert meta["speed"] == pytest.approx(1.25)
    assert meta["expected_duration"] == pytest.approx(2.0)
    assert meta["final_speed_factor"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("speed", "expected_duration"),
    [(0.0, None), (-1.0, None), (float("nan"), None), (1.0, 0.0), (1.0, -1.0)],
)
def test_timing_controls_require_positive_finite_values(speed, expected_duration):
    with pytest.raises(ValueError, match="must be greater than 0"):
        apply_timing_control(tone(), 16000, speed=speed, expected_duration=expected_duration)


def test_sample_rate_must_be_positive():
    with pytest.raises(ValueError, match="sample_rate must be greater than 0"):
        apply_timing_control(tone(), 0)
