import pytest

from src.evaluate import compute_ces, quality_score


def test_quality_score():
    assert quality_score(1.0, 1.0) == pytest.approx(1.0)


def test_quality_score_uses_required_weights():
    expected = 0.70 * 0.8 + 0.30 * 0.6

    assert quality_score(0.8, 0.6) == pytest.approx(expected)


def test_ces_is_zero_when_cpu_limit_is_exceeded():
    assert compute_ces(0.9, 50.01, 1000) == 0.0


def test_ces_accounts_for_model_size():
    quality = 0.8
    cpu_ms = 25.0
    size_bytes = 2 * 1024 * 1024

    expected = quality * (1.0 - 0.20 * 25.0 / 50.0 - 0.10 * 0.5)

    assert compute_ces(
        quality,
        cpu_ms,
        size_bytes,
    ) == pytest.approx(expected)


def test_ces_uses_maximum_size_penalty():
    quality = 0.8
    cpu_ms = 0.0

    result = compute_ces(
        quality,
        cpu_ms,
        8 * 1024 * 1024,
    )

    assert result == pytest.approx(quality * 0.9)
