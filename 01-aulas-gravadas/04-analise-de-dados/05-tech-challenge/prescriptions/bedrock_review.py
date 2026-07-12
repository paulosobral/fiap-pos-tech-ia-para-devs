"""CSV/Excel loading, Bedrock (Claude Sonnet) prompt/response handling and
alert generation for the "Prescrições" tab.

Prescription-inconsistency detection (abrupt dose change, potential drug
interaction, dose change without apparent clinical justification) is pure
LLM semantic reasoning over a small, per-patient prescription history — no
statistical/anomaly model is trained for this capability (design.md D6).
The synthetic dataset is small and there is no appropriate public source,
so review is delegated entirely to AWS Bedrock's Claude Sonnet via the
``Converse`` API.

Every AWS call in this module is wrapped so that ``ClientError``,
``BotoCoreError`` and response-parsing failures are all normalized into
``PrescriptionReviewError`` — a clean, catchable exception the
"Prescrições" tab can display via ``st.error`` without crashing the rest
of the app (same pattern as ``audio.aws_speech.AudioProcessingError``).

Spec: openspec/changes/monitoramento-multimodal-pacientes/specs/prescription-review/spec.md
"""
import json
import re
from typing import Any, Dict, List, Optional

import pandas as pd
from botocore.exceptions import BotoCoreError, ClientError

from alerts.feed import Alert, add_alert

ORIGIN = "Prescrições"

# Required columns for an uploaded prescription-history file, per the
# prescription-review spec ("paciente, medicamento, dose e data").
REQUIRED_COLUMNS = ["paciente", "medicamento", "dose", "data"]

# Region/profile/model chosen after probing real Bedrock access (see task
# report for full details of what was tried):
# - Profile "bedrock" (assumed role bedrock-user-personal-role) has
#   model-invoke permissions, but only on-demand throughput in
#   "us-east-1" is blocked pending an Anthropic use-case form; regions
#   "us-west-2"/"us-east-2" work today with the cross-region inference
#   profile id below.
# - Model id must be an *inference profile* id (prefixed "us."), not the
#   bare foundation-model id — Bedrock rejects on-demand invocation of
#   Claude models without one ("Invocation of model ID ... with
#   on-demand throughput isn't supported").
AWS_REGION = "us-east-2"
BEDROCK_MODEL_ID = "us.anthropic.claude-sonnet-5"
BEDROCK_PROFILE_NAME = "bedrock"


class PrescriptionValidationError(ValueError):
    """Raised when an uploaded prescription file is invalid (unreadable
    format or missing a required column)."""


class PrescriptionReviewError(Exception):
    """Raised for any AWS Bedrock failure, timeout, or unparseable
    response.

    The "Prescrições" tab catches this and shows a clean ``st.error``
    message without letting the exception propagate and crash the rest
    of the Streamlit app / other tabs.
    """


# ── boto3 client construction ────────────────────────────────────────


def build_bedrock_client(profile_name: str = BEDROCK_PROFILE_NAME, region_name: str = AWS_REGION):
    """Build the boto3 Bedrock Runtime client used by this module.

    Kept as a thin wrapper (rather than each caller importing ``boto3``
    directly) so the profile/region choice documented in the module
    docstring lives in exactly one place.

    Args:
        profile_name: AWS named profile to use (the "bedrock" profile
            assumes the ``bedrock-user-personal-role`` IAM role, per the
            task brief).
        region_name: AWS region for the client.

    Returns:
        A ``boto3`` ``bedrock-runtime`` client instance.
    """
    import boto3

    session = boto3.Session(profile_name=profile_name)
    return session.client("bedrock-runtime", region_name=region_name)


# ── CSV/Excel loading + validation ───────────────────────────────────


def load_prescriptions(file, filename: str) -> pd.DataFrame:
    """Load and validate an uploaded prescription-history file.

    Accepts CSV or Excel (``.xlsx``/``.xls``), decided by ``filename``'s
    extension. Column names are matched case-insensitively and
    normalized to the lowercase names in ``REQUIRED_COLUMNS``.

    Args:
        file: Path, file-like object or buffer accepted by
            ``pandas.read_csv``/``pandas.read_excel`` (e.g. a Streamlit
            ``UploadedFile``).
        filename: Original filename, used only to pick CSV vs. Excel
            parsing based on its extension.

    Returns:
        DataFrame with exactly the columns in ``REQUIRED_COLUMNS``
        (lowercase), in that order.

    Raises:
        PrescriptionValidationError: If the file cannot be parsed as
            CSV/Excel, or is missing one of ``REQUIRED_COLUMNS``.
    """
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    try:
        if extension in ("xlsx", "xls"):
            df = pd.read_excel(file)
        else:
            df = pd.read_csv(file)
    except Exception as exc:
        raise PrescriptionValidationError(
            f"Não foi possível ler o arquivo como CSV ou Excel: {exc}"
        ) from exc

    normalized_columns = {col: col.strip().lower() for col in df.columns}
    df = df.rename(columns=normalized_columns)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise PrescriptionValidationError(
            "Arquivo de prescrições sem a(s) coluna(s) obrigatória(s): "
            f"{', '.join(missing)}. Colunas obrigatórias: {', '.join(REQUIRED_COLUMNS)}."
        )

    return df[REQUIRED_COLUMNS].copy()


# ── Bedrock prompt construction ──────────────────────────────────────


def build_review_prompt(patient_df: pd.DataFrame, patient_name: str) -> str:
    """Build a structured prompt asking Claude Sonnet to review one
    patient's prescription history for inconsistencies.

    Args:
        patient_df: Prescription rows for a single patient (columns:
            ``paciente``, ``medicamento``, ``dose``, ``data``).
        patient_name: Name of the patient being reviewed (used only for
            the prompt text; ``patient_df`` should already be filtered).

    Returns:
        The full prompt string to send as the Bedrock ``Converse`` user
        message.
    """
    rows_text = "\n".join(
        f"- Data: {row['data']}, Medicamento: {row['medicamento']}, Dose: {row['dose']}"
        for _, row in patient_df.iterrows()
    )

    return (
        "Você é um assistente clínico que revisa históricos de prescrição "
        "médica em busca de inconsistências. Analise o histórico de "
        f"prescrições do paciente \"{patient_name}\" abaixo, em ordem "
        "cronológica:\n\n"
        f"{rows_text}\n\n"
        "Identifique inconsistências dos seguintes tipos:\n"
        "1. mudanca_dose_abrupta — mudança abrupta de dose de um mesmo "
        "medicamento, sem titulação gradual aparente.\n"
        "2. interacao_medicamentosa — combinação de medicamentos "
        "prescritos no histórico com risco conhecido de interação.\n"
        "3. dose_sem_justificativa — alteração de dose sem justificativa "
        "clínica aparente a partir dos dados fornecidos.\n\n"
        "Responda ESTRITAMENTE em JSON, como uma lista (array) de objetos, "
        "cada um com os campos \"tipo\" (um dos três valores acima) e "
        "\"explicacao\" (texto curto explicando a inconsistência "
        "encontrada). Se nenhuma inconsistência for encontrada, responda "
        "com uma lista vazia: []. Não inclua nenhum texto fora do JSON."
    )


# ── Bedrock response parsing ─────────────────────────────────────────

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def parse_bedrock_response(raw_text: str) -> List[Dict[str, Any]]:
    """Parse Claude's response text into a list of finding dicts.

    Bedrock/Claude is asked to respond with pure JSON, but this parses
    defensively: if the JSON array is embedded in surrounding prose, the
    first ``[...]`` block found is extracted and parsed.

    Args:
        raw_text: Raw text content of the Bedrock response.

    Returns:
        List of finding dicts (each with at least ``tipo`` and
        ``explicacao`` keys, as requested by the prompt). Empty list
        when Claude reports no inconsistencies.

    Raises:
        PrescriptionReviewError: If no valid JSON array can be found/
            parsed in ``raw_text``.
    """
    match = _JSON_ARRAY_RE.search(raw_text)
    if not match:
        raise PrescriptionReviewError(
            f"Resposta do Bedrock não contém um array JSON válido: {raw_text!r}"
        )

    try:
        findings = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise PrescriptionReviewError(
            f"Não foi possível interpretar a resposta do Bedrock como JSON: {exc}"
        ) from exc

    if not isinstance(findings, list):
        raise PrescriptionReviewError(
            f"Resposta do Bedrock não é uma lista JSON: {findings!r}"
        )

    return findings


# ── Alert generation ─────────────────────────────────────────────────


def generate_alerts_for_findings(findings: List[Dict[str, Any]], patient_name: str) -> List[Alert]:
    """Push one ``Alert`` per Bedrock-identified inconsistency.

    Args:
        findings: Parsed findings from ``parse_bedrock_response``.
        patient_name: Patient the findings refer to (included in the
            alert description).

    Returns:
        List of ``Alert`` objects created (also pushed to the shared
        feed). Empty list when ``findings`` is empty.
    """
    alerts = []
    for finding in findings:
        tipo = finding.get("tipo", "inconsistencia")
        explicacao = finding.get("explicacao", "sem detalhes fornecidos.")
        description = f"[{patient_name}] Inconsistência de prescrição ({tipo}): {explicacao}"
        alerts.append(add_alert(origin=ORIGIN, description=description))

    return alerts


# ── End-to-end review (Bedrock call + parsing + alerting) ───────────


def review_patient_prescriptions(
    df: pd.DataFrame,
    patient_name: str,
    client=None,
    model_id: str = BEDROCK_MODEL_ID,
    max_tokens: int = 1024,
) -> Dict[str, Any]:
    """Review one patient's prescription history via AWS Bedrock.

    Filters ``df`` to ``patient_name``, builds the structured prompt,
    calls Bedrock's ``Converse`` API, parses the response and generates
    one ``Alert`` per inconsistency found.

    Args:
        df: Full prescriptions DataFrame (as returned by
            ``load_prescriptions``), covering possibly multiple
            patients.
        patient_name: Patient to review (must match a value in the
            ``paciente`` column).
        client: A ``boto3`` ``bedrock-runtime`` client (or compatible
            mock). Built via ``build_bedrock_client()`` when omitted.
        model_id: Bedrock model/inference-profile id to invoke.
        max_tokens: Max tokens for Claude's response.

    Returns:
        Dict with ``findings`` (list of finding dicts) and ``alerts``
        (list of ``Alert`` objects pushed to the shared feed).

    Raises:
        PrescriptionReviewError: On any Bedrock failure/timeout or
            unparseable response.
    """
    if client is None:
        client = build_bedrock_client()

    patient_df = df[df["paciente"] == patient_name]
    prompt = build_review_prompt(patient_df, patient_name)

    try:
        response = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": max_tokens},
        )
    except (ClientError, BotoCoreError) as exc:
        raise PrescriptionReviewError(
            f"Falha ao chamar o AWS Bedrock para revisão de prescrições: {exc}"
        ) from exc

    try:
        content_blocks = response["output"]["message"]["content"]
        # Some models (e.g. extended-thinking-capable Claude models) emit
        # a leading "reasoningContent" block before the actual "text"
        # answer block, so the first block with a "text" key is used
        # rather than assuming it is always at index 0.
        raw_text = next(block["text"] for block in content_blocks if "text" in block)
    except (KeyError, IndexError, TypeError, StopIteration) as exc:
        raise PrescriptionReviewError(
            f"Resposta inesperada do AWS Bedrock (formato não reconhecido): {exc}"
        ) from exc

    findings = parse_bedrock_response(raw_text)
    alerts = generate_alerts_for_findings(findings, patient_name)

    return {"findings": findings, "alerts": alerts}
