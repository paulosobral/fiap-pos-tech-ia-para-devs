"""Tests for audio/aws_speech.py — AWS Transcribe/Comprehend wrappers and
critical-term detection, all exercised against mocked boto3 clients.

Covers scenarios from the audio-speech-analysis spec:
- Requirement: Upload e transcrição de áudio de consulta
- Requirement: Análise de sentimento e termos críticos via AWS Comprehend
  - Scenario: Texto transcrito contém termo crítico
  - Scenario: AWS Comprehend retorna sentimento negativo

No real AWS call is made in this file — every boto3 client is a
``unittest.mock.MagicMock`` injected via the ``s3_client``/
``transcribe_client``/``client`` parameters. The one real Transcribe call
and one real Comprehend call (per the AWS validation budget) are run
manually/separately, not as part of the automated suite.
"""
from unittest.mock import MagicMock

import pytest
import streamlit as st
from botocore.exceptions import ClientError

from audio.aws_speech import (
    DEFAULT_CRITICAL_TERMS,
    AudioProcessingError,
    analyze_sentiment,
    detect_entities,
    find_critical_terms,
    parse_transcript_items,
    raise_critical_term_alerts,
    transcribe_audio,
)


@pytest.fixture(autouse=True)
def clean_session_state():
    st.session_state.clear()
    yield
    st.session_state.clear()


# ── find_critical_terms ──────────────────────────────────────────────


def test_find_critical_terms_matches_configured_term_case_insensitively():
    text = "Doutor, estou com muita DOR no peito."

    matches = find_critical_terms(text, terms=["dor"])

    assert len(matches) == 1
    assert matches[0]["term"] == "dor"


def test_find_critical_terms_returns_context_snippet_around_match():
    text = "Eu não consigo respirar direito desde ontem à noite."

    matches = find_critical_terms(text, terms=["não consigo respirar"], context_chars=10)

    assert len(matches) == 1
    assert "não consigo respirar" in matches[0]["context"].lower()


def test_find_critical_terms_returns_empty_list_when_no_term_matches():
    text = "Estou me sentindo bem, sem queixas hoje."

    matches = find_critical_terms(text, terms=DEFAULT_CRITICAL_TERMS)

    assert matches == []


def test_find_critical_terms_finds_multiple_distinct_terms():
    text = "Sinto dor no peito e não consigo respirar bem."

    matches = find_critical_terms(text, terms=["dor", "não consigo respirar"])

    matched_terms = {m["term"] for m in matches}
    assert matched_terms == {"dor", "não consigo respirar"}


# ── raise_critical_term_alerts ───────────────────────────────────────


def test_raise_critical_term_alerts_adds_one_alert_per_match():
    text = "Sinto muita dor no peito."

    alerts = raise_critical_term_alerts(text, terms=["dor"])

    assert len(alerts) == 1
    assert alerts[0].origin == "Áudio"
    assert "dor" in alerts[0].description


def test_raise_critical_term_alerts_adds_nothing_when_no_term_matches():
    text = "Tudo bem, sem sintomas."

    alerts = raise_critical_term_alerts(text, terms=DEFAULT_CRITICAL_TERMS)

    assert alerts == []


def test_raise_critical_term_alerts_assigns_sequential_ids():
    # Structured id (change alertas-estruturados-audio-prescricoes): each
    # match gets a unique #A<NN> id, starting at 1 by default.
    text = "Sinto muita dor no peito e também tontura."

    alerts = raise_critical_term_alerts(text, terms=["dor", "tontura"])

    assert [a.alert_id for a in alerts] == ["#A01", "#A02"]


def test_raise_critical_term_alerts_honors_custom_start_index():
    text = "Sinto muita dor no peito."

    alerts = raise_critical_term_alerts(text, terms=["dor"], start_index=5)

    assert alerts[0].alert_id == "#A05"


def test_raise_critical_term_alerts_sets_structured_category():
    # Structured fields (change alertas-estruturados-e-vinculo-sinais-vitais):
    # critical-term alerts now carry a "Termo crítico" category, keeping the
    # description text unchanged for the user.
    text = "Sinto muita dor no peito."

    alerts = raise_critical_term_alerts(text, terms=["dor"])

    assert len(alerts) == 1
    assert alerts[0].category == "Termo crítico"
    assert "dor" in alerts[0].description


# ── parse_transcript_items ───────────────────────────────────────────


def test_parse_transcript_items_keeps_only_pronunciation_items_with_timestamps():
    items = [
        {
            "type": "pronunciation",
            "start_time": "0.0",
            "end_time": "0.5",
            "alternatives": [{"content": "Doutor"}],
        },
        {"type": "punctuation", "alternatives": [{"content": ","}]},
        {
            "type": "pronunciation",
            "start_time": "0.6",
            "end_time": "1.0",
            "alternatives": [{"content": "dor"}],
        },
    ]

    words = parse_transcript_items(items)

    assert len(words) == 2
    assert words[0] == {"content": "Doutor", "start_time": 0.0, "end_time": 0.5}
    assert words[1] == {"content": "dor", "start_time": 0.6, "end_time": 1.0}


def test_parse_transcript_items_returns_empty_list_for_no_items():
    assert parse_transcript_items([]) == []


# ── transcribe_audio (mocked S3 + Transcribe clients) ────────────────


def _completed_job_status(job_name, bucket_name):
    return {
        "TranscriptionJob": {
            "TranscriptionJobName": job_name,
            "TranscriptionJobStatus": "COMPLETED",
            "Transcript": {"TranscriptFileUri": f"https://example.com/{job_name}.json"},
        }
    }


def _transcript_json_body(text, items):
    import json
    import io

    payload = json.dumps(
        {"results": {"transcripts": [{"transcript": text}], "items": items}}
    ).encode("utf-8")

    class _FakeBody:
        def read(self):
            return payload

    return {"Body": _FakeBody()}


def test_transcribe_audio_orchestrates_upload_start_poll_and_fetch():
    s3_client = MagicMock()
    transcribe_client = MagicMock()

    job_status_in_progress = {
        "TranscriptionJob": {"TranscriptionJobStatus": "IN_PROGRESS"}
    }
    job_status_completed = _completed_job_status("job-123", "my-bucket")
    transcribe_client.get_transcription_job.side_effect = [
        job_status_in_progress,
        job_status_completed,
    ]

    items = [
        {
            "type": "pronunciation",
            "start_time": "0.0",
            "end_time": "0.4",
            "alternatives": [{"content": "Doutor"}],
        }
    ]
    s3_client.get_object.return_value = _transcript_json_body("Doutor, sinto dor.", items)

    result = transcribe_audio(
        audio_bytes=b"fake-audio-bytes",
        file_extension="mp3",
        bucket_name="my-bucket",
        job_name="job-123",
        s3_client=s3_client,
        transcribe_client=transcribe_client,
        poll_interval_s=0,
        timeout_s=10,
        sleep_fn=lambda _seconds: None,
    )

    assert result["text"] == "Doutor, sinto dor."
    assert result["words"] == [{"content": "Doutor", "start_time": 0.0, "end_time": 0.4}]
    s3_client.put_object.assert_called_once()
    transcribe_client.start_transcription_job.assert_called_once()
    assert transcribe_client.get_transcription_job.call_count == 2


def test_transcribe_audio_raises_clean_error_when_job_fails():
    s3_client = MagicMock()
    transcribe_client = MagicMock()
    transcribe_client.get_transcription_job.return_value = {
        "TranscriptionJob": {
            "TranscriptionJobStatus": "FAILED",
            "FailureReason": "Unsupported media format",
        }
    }

    with pytest.raises(AudioProcessingError, match="Unsupported media format"):
        transcribe_audio(
            audio_bytes=b"fake",
            file_extension="mp3",
            bucket_name="my-bucket",
            job_name="job-fail",
            s3_client=s3_client,
            transcribe_client=transcribe_client,
            poll_interval_s=0,
            timeout_s=10,
            sleep_fn=lambda _seconds: None,
        )


def test_transcribe_audio_raises_clean_error_on_timeout():
    s3_client = MagicMock()
    transcribe_client = MagicMock()
    transcribe_client.get_transcription_job.return_value = {
        "TranscriptionJob": {"TranscriptionJobStatus": "IN_PROGRESS"}
    }

    fake_clock = {"now": 0.0}

    def fake_time_fn():
        return fake_clock["now"]

    def fake_sleep(seconds):
        fake_clock["now"] += seconds + 1

    with pytest.raises(AudioProcessingError, match="[Tt]empo limite"):
        transcribe_audio(
            audio_bytes=b"fake",
            file_extension="mp3",
            bucket_name="my-bucket",
            job_name="job-timeout",
            s3_client=s3_client,
            transcribe_client=transcribe_client,
            poll_interval_s=1,
            timeout_s=2,
            sleep_fn=fake_sleep,
            time_fn=fake_time_fn,
        )


def test_transcribe_audio_raises_clean_error_on_boto_client_error():
    s3_client = MagicMock()
    s3_client.put_object.side_effect = ClientError(
        {"Error": {"Code": "AccessDenied", "Message": "no permission"}}, "PutObject"
    )
    transcribe_client = MagicMock()

    with pytest.raises(AudioProcessingError):
        transcribe_audio(
            audio_bytes=b"fake",
            file_extension="mp3",
            bucket_name="my-bucket",
            job_name="job-error",
            s3_client=s3_client,
            transcribe_client=transcribe_client,
            poll_interval_s=0,
            timeout_s=10,
            sleep_fn=lambda _seconds: None,
        )


# ── analyze_sentiment / detect_entities (mocked Comprehend client) ───


def test_analyze_sentiment_returns_sentiment_label_and_scores():
    client = MagicMock()
    client.detect_sentiment.return_value = {
        "Sentiment": "NEGATIVE",
        "SentimentScore": {"Positive": 0.01, "Negative": 0.9, "Neutral": 0.05, "Mixed": 0.04},
    }

    result = analyze_sentiment("Sinto muita dor.", client=client)

    assert result["sentiment"] == "NEGATIVE"
    assert result["sentiment_score"]["Negative"] == 0.9
    client.detect_sentiment.assert_called_once()


def test_analyze_sentiment_raises_clean_error_on_client_error():
    client = MagicMock()
    client.detect_sentiment.side_effect = ClientError(
        {"Error": {"Code": "Throttling", "Message": "too many requests"}}, "DetectSentiment"
    )

    with pytest.raises(AudioProcessingError):
        analyze_sentiment("texto qualquer", client=client)


def test_detect_entities_returns_entities_list():
    client = MagicMock()
    client.detect_entities.return_value = {
        "Entities": [{"Text": "peito", "Type": "OTHER", "Score": 0.8}]
    }

    entities = detect_entities("Sinto dor no peito.", client=client)

    assert entities == [{"Text": "peito", "Type": "OTHER", "Score": 0.8}]


def test_detect_entities_raises_clean_error_on_client_error():
    client = MagicMock()
    client.detect_entities.side_effect = ClientError(
        {"Error": {"Code": "Throttling", "Message": "too many requests"}}, "DetectEntities"
    )

    with pytest.raises(AudioProcessingError):
        detect_entities("texto qualquer", client=client)
