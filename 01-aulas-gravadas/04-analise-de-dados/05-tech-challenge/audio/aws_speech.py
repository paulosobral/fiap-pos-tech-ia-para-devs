"""AWS Transcribe + Comprehend wrappers and critical-term detection for
the "Áudio" tab.

AWS Transcribe is asynchronous: a job is started against an audio file
sitting in S3, and the caller polls ``GetTranscriptionJob`` until it
reaches a terminal state, then fetches the resulting JSON transcript
(also from S3, at the URI Transcribe wrote the output to). This module
wraps that whole flow in ``transcribe_audio``, and wraps the synchronous
AWS Comprehend ``DetectSentiment``/``DetectEntities`` calls in
``analyze_sentiment``/``detect_entities``.

Region choice (documented per Global Constraints / task brief): callers
should construct the ``boto3`` clients passed into this module's
functions with ``region_name="us-east-1"`` rather than relying on the
``default`` profile's configured region (``sa-east-1``), because
Transcribe/Comprehend availability in ``sa-east-1`` was not verified for
this task and ``us-east-1`` is definitely supported for both.

Every AWS call in this module is wrapped so that ``ClientError``,
``BotoCoreError`` and a client-side polling timeout are all normalized
into ``AudioProcessingError`` — a clean, catchable exception the "Áudio"
tab can display via ``st.error`` without crashing the rest of the app
(same pattern the Bedrock/prescriptions tab will need later).

Spec: openspec/changes/monitoramento-multimodal-pacientes/specs/audio-speech-analysis/spec.md
"""
import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence

from botocore.exceptions import BotoCoreError, ClientError

from alerts.feed import Alert, add_alert

ORIGIN = "Áudio"

# Region used for the boto3 Transcribe/Comprehend/S3 clients built by
# ``build_clients`` below — see module docstring for why this differs
# from the ``default`` AWS profile's configured region (sa-east-1).
AWS_REGION = "us-east-1"

# Name of the environment variable the "Áudio" tab reads to pick up a
# pre-existing S3 bucket for Transcribe's input/output objects (task
# brief: Transcribe needs an S3 bucket; this project does not keep one
# provisioned permanently — see README/report for setup instructions).
AUDIO_S3_BUCKET_ENV_VAR = "AUDIO_TRANSCRIBE_BUCKET"


def build_clients(region_name: str = AWS_REGION) -> Dict[str, Any]:
    """Build the three boto3 clients this module's functions need.

    Kept as a thin wrapper (rather than each caller importing ``boto3``
    directly) so the region choice documented in the module docstring
    lives in exactly one place.

    Args:
        region_name: AWS region for all three clients.

    Returns:
        Dict with ``s3``, ``transcribe`` and ``comprehend`` boto3 client
        instances.
    """
    import boto3

    return {
        "s3": boto3.client("s3", region_name=region_name),
        "transcribe": boto3.client("transcribe", region_name=region_name),
        "comprehend": boto3.client("comprehend", region_name=region_name),
    }


def get_configured_bucket_name() -> Optional[str]:
    """Return the S3 bucket name configured via ``AUDIO_S3_BUCKET_ENV_VAR``.

    Returns:
        The bucket name, or ``None`` if the environment variable is not
        set (the "Áudio" tab shows a clean setup message in that case
        instead of attempting a Transcribe call that would fail).
    """
    return os.environ.get(AUDIO_S3_BUCKET_ENV_VAR) or None

# Configurable list of critical terms to look for in the transcribed
# text (task 5.4). Kept as a plain module-level list so it's trivial to
# extend/override per deployment without touching function signatures.
DEFAULT_CRITICAL_TERMS: List[str] = [
    "dor",
    "não consigo respirar",
    "nao consigo respirar",
    "falta de ar",
    "dor no peito",
    "tontura",
    "desmaio",
]


class AudioProcessingError(Exception):
    """Raised for any AWS Transcribe/Comprehend failure or timeout.

    The "Áudio" tab catches this and shows a clean ``st.error`` message
    without letting the exception propagate and crash the rest of the
    Streamlit app / other tabs.
    """


# ── Critical-term detection ──────────────────────────────────────────


def find_critical_terms(
    text: str, terms: Sequence[str] = DEFAULT_CRITICAL_TERMS, context_chars: int = 40
) -> List[Dict[str, str]]:
    """Find configured critical terms in ``text`` (case-insensitive).

    Args:
        text: Transcribed text to search.
        terms: Critical terms to look for (configurable list).
        context_chars: Number of characters of context to include on
            each side of the match.

    Returns:
        List of dicts, one per match, each with ``term`` (the configured
        term that matched) and ``context`` (a snippet of ``text``
        surrounding the match, for display/alerting).
    """
    matches: List[Dict[str, str]] = []
    lower_text = text.lower()

    for term in terms:
        lower_term = term.lower()
        start = 0
        while True:
            index = lower_text.find(lower_term, start)
            if index == -1:
                break
            context_start = max(0, index - context_chars)
            context_end = min(len(text), index + len(term) + context_chars)
            matches.append({"term": term, "context": text[context_start:context_end].strip()})
            start = index + len(lower_term)

    return matches


def raise_critical_term_alerts(
    text: str, terms: Sequence[str] = DEFAULT_CRITICAL_TERMS, context_chars: int = 40
) -> List[Alert]:
    """Find critical terms in ``text`` and push one ``Alert`` per match.

    Args:
        text: Transcribed text to search.
        terms: Critical terms to look for.
        context_chars: Context window size passed to ``find_critical_terms``.

    Returns:
        List of ``Alert`` objects created (also pushed to the shared feed).
    """
    matches = find_critical_terms(text, terms=terms, context_chars=context_chars)

    alerts = []
    for match in matches:
        description = (
            f"Termo crítico detectado na transcrição: \"{match['term']}\" "
            f"— contexto: \"{match['context']}\"."
        )
        # Structured category (change alertas-estruturados-e-vinculo-sinais-
        # vitais) so a later export reads a clean column; text unchanged.
        alerts.append(
            add_alert(origin=ORIGIN, description=description, category="Termo crítico")
        )

    return alerts


# ── Transcript parsing ───────────────────────────────────────────────


def parse_transcript_items(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract per-word timestamps from AWS Transcribe's ``items`` list.

    AWS Transcribe's result JSON mixes ``"pronunciation"`` items (actual
    spoken words, with ``start_time``/``end_time``) and
    ``"punctuation"`` items (no timestamps) in the same ``items`` array.
    Only pronunciation items carry the timing data needed for the
    speech-rate/pause-duration derivation in ``audio/analysis.py``.

    Args:
        items: Raw ``results.items`` list from a Transcribe transcript
            JSON.

    Returns:
        List of dicts with ``content`` (str), ``start_time`` (float,
        seconds) and ``end_time`` (float, seconds), one per spoken word,
        in original order.
    """
    words = []
    for item in items:
        if item.get("type") != "pronunciation":
            continue
        words.append(
            {
                "content": item["alternatives"][0]["content"],
                "start_time": float(item["start_time"]),
                "end_time": float(item["end_time"]),
            }
        )
    return words


# ── AWS Transcribe (async job) ───────────────────────────────────────

_TERMINAL_STATUSES = {"COMPLETED", "FAILED"}


def transcribe_audio(
    audio_bytes: bytes,
    file_extension: str,
    bucket_name: str,
    s3_client,
    transcribe_client,
    job_name: Optional[str] = None,
    language_code: str = "pt-BR",
    poll_interval_s: float = 5.0,
    timeout_s: float = 300.0,
    sleep_fn=time.sleep,
    time_fn=time.monotonic,
) -> Dict[str, Any]:
    """Transcribe ``audio_bytes`` via AWS Transcribe, returning text + words.

    Flow: upload the audio to ``bucket_name`` in S3, start a Transcribe
    job pointing at that S3 object, poll ``GetTranscriptionJob`` until it
    reaches a terminal state (or ``timeout_s`` elapses), then fetch and
    parse the resulting transcript JSON (also stored in S3).

    Args:
        audio_bytes: Raw audio file content.
        file_extension: File extension without the dot (``"mp3"`` or
            ``"wav"``), used both as the S3 object suffix and the
            ``MediaFormat`` passed to Transcribe.
        bucket_name: S3 bucket used for both input audio and Transcribe
            output. Must already exist and be accessible to the caller's
            credentials.
        s3_client: A ``boto3`` S3 client (or compatible mock).
        transcribe_client: A ``boto3`` Transcribe client (or compatible
            mock).
        job_name: Optional explicit Transcribe job name; a random UUID
            is used when omitted (job names must be unique per account).
        language_code: Transcribe language code (defaults to Brazilian
            Portuguese, matching the demo audio).
        poll_interval_s: Seconds to sleep between polling attempts.
        timeout_s: Maximum seconds to wait for the job to finish before
            raising ``AudioProcessingError``.
        sleep_fn: Injectable sleep function (tests pass a no-op).
        time_fn: Injectable monotonic clock function (tests pass a fake
            clock to exercise the timeout path deterministically).

    Returns:
        Dict with ``text`` (full transcript string) and ``words`` (list
        of per-word dicts as returned by ``parse_transcript_items``).

    Raises:
        AudioProcessingError: On any AWS error, a ``FAILED`` job status,
            or if the job does not reach a terminal state within
            ``timeout_s``.
    """
    job_name = job_name or f"tech-challenge-audio-{uuid.uuid4().hex}"
    input_key = f"audio-input/{job_name}.{file_extension}"
    output_key = f"{job_name}.json"

    try:
        s3_client.put_object(Bucket=bucket_name, Key=input_key, Body=audio_bytes)

        media_format = "mp3" if file_extension.lower() == "mp3" else "wav"
        transcribe_client.start_transcription_job(
            TranscriptionJobName=job_name,
            LanguageCode=language_code,
            MediaFormat=media_format,
            Media={"MediaFileUri": f"s3://{bucket_name}/{input_key}"},
            OutputBucketName=bucket_name,
            OutputKey=output_key,
        )

        start = time_fn()
        while True:
            response = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
            job = response["TranscriptionJob"]
            status = job["TranscriptionJobStatus"]

            if status == "FAILED":
                reason = job.get("FailureReason", "motivo não informado")
                raise AudioProcessingError(
                    f"Job de transcrição falhou: {reason}"
                )
            if status == "COMPLETED":
                break

            if time_fn() - start >= timeout_s:
                raise AudioProcessingError(
                    f"Tempo limite excedido aguardando a transcrição (job '{job_name}')."
                )

            sleep_fn(poll_interval_s)

        transcript_object = s3_client.get_object(Bucket=bucket_name, Key=output_key)
        transcript_json = json.loads(transcript_object["Body"].read())

        results = transcript_json["results"]
        text = results["transcripts"][0]["transcript"]
        words = parse_transcript_items(results.get("items", []))

        _cleanup_s3_objects(s3_client, bucket_name, [input_key, output_key])
        _cleanup_transcription_job(transcribe_client, job_name)

        return {"text": text, "words": words}

    except AudioProcessingError:
        raise
    except (ClientError, BotoCoreError) as exc:
        raise AudioProcessingError(f"Falha ao comunicar com AWS Transcribe/S3: {exc}") from exc


def _cleanup_s3_objects(s3_client, bucket_name: str, keys: List[str]) -> None:
    """Best-effort deletion of the temporary input/output S3 objects.

    Runs after a successful transcription so repeated real usage of the
    "Áudio" tab does not accumulate small files in the configured
    bucket. Failures here are swallowed (logged as a no-op) — they must
    never mask an otherwise-successful transcription result.
    """
    for key in keys:
        try:
            s3_client.delete_object(Bucket=bucket_name, Key=key)
        except (ClientError, BotoCoreError):
            pass


def _cleanup_transcription_job(transcribe_client, job_name: str) -> None:
    """Best-effort deletion of the Transcribe job metadata itself."""
    try:
        transcribe_client.delete_transcription_job(TranscriptionJobName=job_name)
    except (ClientError, BotoCoreError):
        pass


# ── AWS Comprehend (synchronous) ─────────────────────────────────────


def analyze_sentiment(text: str, client, language_code: str = "pt") -> Dict[str, Any]:
    """Classify sentiment of ``text`` via AWS Comprehend ``DetectSentiment``.

    Args:
        text: Text to analyze (e.g. the transcript from ``transcribe_audio``).
        client: A ``boto3`` Comprehend client (or compatible mock).
        language_code: Comprehend language code.

    Returns:
        Dict with ``sentiment`` (one of ``POSITIVE``/``NEGATIVE``/
        ``NEUTRAL``/``MIXED``) and ``sentiment_score`` (dict of
        confidence scores per label).

    Raises:
        AudioProcessingError: On any AWS Comprehend failure.
    """
    try:
        response = client.detect_sentiment(Text=text, LanguageCode=language_code)
        return {"sentiment": response["Sentiment"], "sentiment_score": response["SentimentScore"]}
    except (ClientError, BotoCoreError) as exc:
        raise AudioProcessingError(f"Falha ao chamar AWS Comprehend (sentimento): {exc}") from exc


def detect_entities(text: str, client, language_code: str = "pt") -> List[Dict[str, Any]]:
    """Extract entities from ``text`` via AWS Comprehend ``DetectEntities``.

    Args:
        text: Text to analyze.
        client: A ``boto3`` Comprehend client (or compatible mock).
        language_code: Comprehend language code.

    Returns:
        List of entity dicts as returned by Comprehend (``Text``,
        ``Type``, ``Score``, ...).

    Raises:
        AudioProcessingError: On any AWS Comprehend failure.
    """
    try:
        response = client.detect_entities(Text=text, LanguageCode=language_code)
        return response["Entities"]
    except (ClientError, BotoCoreError) as exc:
        raise AudioProcessingError(f"Falha ao chamar AWS Comprehend (entidades): {exc}") from exc
