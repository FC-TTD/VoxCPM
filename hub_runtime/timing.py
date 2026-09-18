"""Unified post-generation timing controls for the Hub runtime."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from ttd_model_runtime.audio.speed import time_stretch_wav


def apply_timing_control(
    wav: np.ndarray,
    sample_rate: int,
    *,
    speed: float = 1.0,
    expected_duration: Optional[float] = None,
) -> tuple[np.ndarray, dict[str, Optional[float]]]:
    """Apply speed or target-duration control to a generated waveform.

    ``expected_duration`` takes precedence over ``speed`` when both are given.
    The returned metadata describes the waveform before and after this stage.
    """
    if int(sample_rate) <= 0:
        raise ValueError("sample_rate must be greater than 0")
    speed_value = _positive_finite(speed, "speed")
    expected_value = (
        None
        if expected_duration is None
        else _positive_finite(expected_duration, "expected_duration")
    )

    audio = np.asarray(wav)
    original_duration = float(len(audio)) / float(sample_rate)
    if original_duration <= 0.0:
        if expected_value is not None:
            raise ValueError("expected_duration cannot be applied to empty audio")
        return audio, _metadata(speed_value, None, 0.0, 0.0, speed_value)

    final_speed_factor = (
        original_duration / expected_value
        if expected_value is not None
        else speed_value
    )
    if not math.isfinite(final_speed_factor) or final_speed_factor <= 0.0:
        raise ValueError("computed speed factor must be greater than 0")

    if abs(final_speed_factor - 1.0) < 1e-9:
        stretched = audio
    else:
        # Do not silently claim timing support when SoX or the helper fails.
        stretched = time_stretch_wav(
            audio,
            int(sample_rate),
            float(final_speed_factor),
            allow_passthrough_on_failure=False,
        )

    final_duration = float(len(stretched)) / float(sample_rate)
    return stretched, _metadata(
        speed_value,
        expected_value,
        original_duration,
        final_duration,
        final_speed_factor,
    )


def _positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be greater than 0")
    return result


def _metadata(
    speed: float,
    expected_duration: Optional[float],
    original_duration: float,
    final_duration: float,
    final_speed_factor: float,
) -> dict[str, Optional[float]]:
    return {
        "speed": float(speed),
        "expected_duration": expected_duration,
        "original_duration_seconds": float(original_duration),
        "final_duration_seconds": float(final_duration),
        "final_speed_factor": float(final_speed_factor),
    }
