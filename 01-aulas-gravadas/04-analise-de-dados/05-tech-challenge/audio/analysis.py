"""Acoustic feature derivation and speech-anomaly detection for the
"Áudio" tab.

Derives two series from AWS Transcribe's per-word timestamps (task 5.5):
- Speech rate: words per second, computed per fixed-length segment.
- Pause duration: gap (seconds) between the end of one word and the
  start of the next.

Both series are run through the shared rolling z-score detector
(``anomaly.zscore.detect_anomalies``, task 5.6) to flag segments/gaps
compatible with fatigue or dysarthria — an unusually slow speech rate or
an unusually long pause. Every flagged point raises an ``Alert`` on the
shared feed.

Spec: openspec/changes/monitoramento-multimodal-pacientes/specs/audio-speech-analysis/spec.md
"""
from typing import Any, Dict, List, Sequence

import pandas as pd

from alerts.feed import add_alert
from anomaly.zscore import detect_anomalies

ORIGIN = "Áudio"

# Fixed thresholds for the rolling z-score layer (Global Constraints:
# "áudio e sinais vitais use a fixed threshold documented in code").
DEFAULT_WINDOW = 5
DEFAULT_THRESHOLD = 2.5

# Fixed-length window (seconds) used to bucket words into segments for
# the speech-rate series.
DEFAULT_SEGMENT_SECONDS = 3.0


def segment_words(words: Sequence[Dict[str, Any]], segment_seconds: float = DEFAULT_SEGMENT_SECONDS) -> List[Dict[str, Any]]:
    """Group ``words`` into fixed-length time segments.

    Args:
        words: Per-word dicts as returned by
            ``audio.aws_speech.parse_transcript_items`` (keys
            ``content``, ``start_time``, ``end_time``).
        segment_seconds: Length of each segment, in seconds, measured
            from the first word's ``start_time``.

    Returns:
        List of dicts, one per non-empty segment, each with
        ``segment_start_s``, ``segment_end_s`` and ``words`` (the
        subset of ``words`` whose ``start_time`` falls in that window).
        Empty list when ``words`` is empty.
    """
    if not words:
        return []

    first_start = words[0]["start_time"]
    segments: List[Dict[str, Any]] = []
    current_index = 0

    segment_start = first_start
    while current_index < len(words):
        segment_end = segment_start + segment_seconds
        segment_words_list = [
            w for w in words[current_index:] if w["start_time"] < segment_end
        ]
        if segment_words_list:
            segments.append(
                {
                    "segment_start_s": segment_start,
                    "segment_end_s": segment_end,
                    "words": segment_words_list,
                }
            )
            current_index += len(segment_words_list)
        segment_start = segment_end

    return segments


def derive_speech_rate(segments: Sequence[Dict[str, Any]]) -> pd.Series:
    """Compute words-per-second speech rate for each segment.

    The denominator is each segment's own duration
    (``segment_end_s - segment_start_s``), so this stays consistent
    regardless of the ``segment_seconds`` used to build ``segments``.

    Args:
        segments: Segments as returned by ``segment_words``.

    Returns:
        Series of speech rate (words/second), aligned by position with
        ``segments``. Empty Series when ``segments`` is empty.
    """
    if not segments:
        return pd.Series(dtype="float64")

    rates = [
        len(segment["words"]) / (segment["segment_end_s"] - segment["segment_start_s"])
        for segment in segments
    ]
    return pd.Series(rates, dtype="float64")


def derive_pause_durations(words: Sequence[Dict[str, Any]]) -> pd.Series:
    """Compute the pause (seconds) between each pair of consecutive words.

    Args:
        words: Per-word dicts ordered by time (keys ``start_time``,
            ``end_time``).

    Returns:
        Series of length ``len(words) - 1`` with the gap between each
        word's end and the next word's start. Empty Series when there
        are fewer than two words.
    """
    if len(words) < 2:
        return pd.Series(dtype="float64")

    pauses = [
        words[i + 1]["start_time"] - words[i]["end_time"] for i in range(len(words) - 1)
    ]
    return pd.Series(pauses, dtype="float64")


def _speech_rate_alert_description(rate: float, segment: Dict[str, Any], threshold: float) -> str:
    return (
        f"Taxa de fala anômala em t={segment['segment_start_s']:.2f}s-"
        f"{segment['segment_end_s']:.2f}s: {rate:.2f} palavras/s "
        f"(|z-score| > {threshold}) — possível indicador de fadiga/disartria."
    )


def _pause_alert_description(pause: float, timestamp_s: float, threshold: float) -> str:
    return (
        f"Pausa anômala de {pause:.2f}s em t={timestamp_s:.2f}s "
        f"(|z-score| > {threshold}) — possível indicador de fadiga/disartria."
    )


def analyze(
    words: Sequence[Dict[str, Any]],
    window: int = DEFAULT_WINDOW,
    threshold: float = DEFAULT_THRESHOLD,
    segment_seconds: float = DEFAULT_SEGMENT_SECONDS,
) -> Dict[str, Any]:
    """Derive acoustic feature series from ``words`` and flag anomalies.

    Args:
        words: Per-word dicts as returned by
            ``audio.aws_speech.parse_transcript_items``.
        window: Rolling window size for the z-score layer.
        threshold: Z-score magnitude above which a reading is anomalous.
        segment_seconds: Segment length used for the speech-rate series.

    Returns:
        Dict with:
            - ``speech_rate``: Series of words/second per segment.
            - ``pause_durations``: Series of pause (seconds) between
              consecutive words.
            - ``speech_rate_anomalies`` / ``pause_anomalies``: boolean
              Series aligned with the above, from
              ``anomaly.zscore.detect_anomalies``.
            - ``alerts``: list of ``Alert`` objects generated for every
              flagged segment/pause (also pushed to the shared feed).

    Never raises on empty/short ``words`` — mirrors
    ``detect_anomalies``'s "series shorter than window -> all False"
    contract.
    """
    segments = segment_words(words, segment_seconds=segment_seconds)
    speech_rate = derive_speech_rate(segments)
    pause_durations = derive_pause_durations(words)

    speech_rate_anomalies = detect_anomalies(speech_rate, window=window, threshold=threshold)
    pause_anomalies = detect_anomalies(pause_durations, window=window, threshold=threshold)

    alerts = []

    for idx in speech_rate_anomalies[speech_rate_anomalies].index:
        rate = speech_rate.iloc[idx]
        segment = segments[idx]
        description = _speech_rate_alert_description(rate, segment, threshold)
        alerts.append(add_alert(origin=ORIGIN, description=description))

    for idx in pause_anomalies[pause_anomalies].index:
        pause = pause_durations.iloc[idx]
        # Pause i is the gap between words[i] and words[i+1]; report the
        # timestamp of the pause's start (end of the earlier word).
        timestamp_s = words[idx]["end_time"]
        description = _pause_alert_description(pause, timestamp_s, threshold)
        alerts.append(add_alert(origin=ORIGIN, description=description))

    return {
        "speech_rate": speech_rate,
        "pause_durations": pause_durations,
        "speech_rate_anomalies": speech_rate_anomalies,
        "pause_anomalies": pause_anomalies,
        "alerts": alerts,
    }
