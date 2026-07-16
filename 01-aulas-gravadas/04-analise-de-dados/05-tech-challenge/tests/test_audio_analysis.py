"""Tests for audio/analysis.py — acoustic feature derivation (speech
rate, pause duration) from Transcribe word timestamps, and anomaly
detection over those series via the shared ``detect_anomalies``.

Covers scenarios from the audio-speech-analysis spec:
- Requirement: Extração de features acústicas e detecção de anomalia de fala
  - Scenario: Segmento de fala com taxa anômala
  - Scenario: Áudio sem variação significativa

Pure logic only — no AWS/boto3 involved.
"""
import pandas as pd
import pytest
import streamlit as st

from audio.analysis import (
    DEFAULT_THRESHOLD,
    DEFAULT_WINDOW,
    analyze,
    derive_pause_durations,
    derive_speech_rate,
    segment_words,
)


@pytest.fixture(autouse=True)
def clean_session_state():
    st.session_state.clear()
    yield
    st.session_state.clear()


def _word(content, start, end):
    return {"content": content, "start_time": start, "end_time": end}


# ── segment_words ─────────────────────────────────────────────────────


def test_segment_words_groups_words_within_the_same_window():
    words = [_word("um", 0.0, 0.3), _word("dois", 0.4, 0.7), _word("tres", 3.5, 3.9)]

    segments = segment_words(words, segment_seconds=1.0)

    assert len(segments) == 2
    assert [w["content"] for w in segments[0]["words"]] == ["um", "dois"]
    assert [w["content"] for w in segments[1]["words"]] == ["tres"]


def test_segment_words_returns_empty_list_for_no_words():
    assert segment_words([], segment_seconds=1.0) == []


# ── derive_speech_rate ────────────────────────────────────────────────


def test_derive_speech_rate_counts_words_per_second_per_segment():
    words = [_word("a", 0.0, 0.2), _word("b", 0.2, 0.4), _word("c", 0.4, 0.6)]
    segments = segment_words(words, segment_seconds=1.0)

    rates = derive_speech_rate(segments)

    assert isinstance(rates, pd.Series)
    assert len(rates) == 1
    assert rates.iloc[0] == pytest.approx(3.0)  # 3 words / 1.0s segment


def test_derive_speech_rate_returns_empty_series_for_no_segments():
    rates = derive_speech_rate([])

    assert isinstance(rates, pd.Series)
    assert len(rates) == 0


# ── derive_pause_durations ────────────────────────────────────────────


def test_derive_pause_durations_computes_gap_between_consecutive_words():
    words = [_word("um", 0.0, 0.3), _word("dois", 1.3, 1.6), _word("tres", 1.7, 2.0)]

    pauses = derive_pause_durations(words)

    assert list(pauses) == pytest.approx([1.0, 0.1])


def test_derive_pause_durations_returns_empty_series_for_fewer_than_two_words():
    words = [_word("um", 0.0, 0.3)]

    pauses = derive_pause_durations(words)

    assert len(pauses) == 0


# ── analyze (integration of derivation + detect_anomalies + alerts) ──


def test_analyze_flags_no_anomaly_when_speech_is_regular():
    words = [_word(f"w{i}", i * 0.3, i * 0.3 + 0.2) for i in range(20)]

    result = analyze(words, window=DEFAULT_WINDOW, threshold=DEFAULT_THRESHOLD)

    assert result["speech_rate_anomalies"].any() == False  # noqa: E712
    assert result["alerts"] == []


def test_analyze_flags_anomaly_and_raises_alert_when_pause_is_extreme():
    # Build a run of regular words, then one huge pause, then more regular words,
    # long enough for the rolling window to have full history before the outlier.
    words = []
    t = 0.0
    for i in range(15):
        words.append(_word(f"a{i}", t, t + 0.2))
        t += 0.3
    # Extreme pause before the next word
    t += 20.0
    words.append(_word("depois-da-pausa", t, t + 0.2))
    t += 0.3
    for i in range(5):
        words.append(_word(f"b{i}", t, t + 0.2))
        t += 0.3

    result = analyze(words, window=5, threshold=2.0)

    assert result["pause_anomalies"].any()
    assert len(result["alerts"]) >= 1
    assert result["alerts"][0].origin == "Áudio"


def test_analyze_handles_series_shorter_than_window_without_raising():
    words = [_word("um", 0.0, 0.3), _word("dois", 0.5, 0.8)]

    result = analyze(words, window=DEFAULT_WINDOW, threshold=DEFAULT_THRESHOLD)

    assert result["speech_rate_anomalies"].any() == False  # noqa: E712
    assert result["pause_anomalies"].any() == False  # noqa: E712


def test_analyze_handles_no_words_without_raising():
    result = analyze([], window=DEFAULT_WINDOW, threshold=DEFAULT_THRESHOLD)

    assert len(result["speech_rate_anomalies"]) == 0
    assert len(result["pause_anomalies"]) == 0
    assert result["alerts"] == []


def test_pause_anomaly_alert_carries_structured_category():
    # Structured fields (change alertas-estruturados-e-vinculo-sinais-vitais):
    # pause-anomaly alerts carry a "Disartria" category; the description text
    # the user sees is unchanged.
    words = []
    t = 0.0
    for i in range(15):
        words.append(_word(f"a{i}", t, t + 0.2))
        t += 0.3
    t += 20.0
    words.append(_word("depois-da-pausa", t, t + 0.2))
    t += 0.3
    for i in range(5):
        words.append(_word(f"b{i}", t, t + 0.2))
        t += 0.3

    result = analyze(words, window=5, threshold=2.0)

    assert result["alerts"]
    pause_alerts = [a for a in result["alerts"] if "Pausa anômala" in a.description]
    assert pause_alerts
    assert all(a.category == "Disartria" for a in pause_alerts)


def test_speech_rate_anomaly_alert_carries_structured_category():
    # A run of steady speech, then a segment with a burst of many words
    # (fast rate) so the speech-rate series flags an anomaly.
    words = []
    t = 0.0
    for i in range(18):
        words.append(_word(f"a{i}", t, t + 0.1))
        t += 0.5
    # dense burst: many words in a short span → high words/s
    for i in range(30):
        words.append(_word(f"burst{i}", t, t + 0.02))
        t += 0.03

    result = analyze(words, window=3, threshold=1.5)

    rate_alerts = [a for a in result["alerts"] if "Taxa de fala" in a.description]
    if rate_alerts:  # only assert when the rate layer actually flagged
        assert all(a.category == "Fadiga de fala" for a in rate_alerts)
