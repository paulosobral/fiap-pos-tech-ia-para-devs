"""Tests for prescriptions/bedrock_review.py — CSV/Excel loading and
validation, Bedrock prompt/response handling, and alert generation for the
"Prescrições" tab, all exercised against mocked boto3 Bedrock clients.

Covers scenarios from the prescription-review spec:
- Requirement: Upload de histórico de prescrições
  - Scenario: Upload de arquivo de prescrições válido
  - Scenario: Upload de arquivo com colunas faltantes
- Requirement: Análise de inconsistências via AWS Bedrock
  - Scenario: Bedrock identifica inconsistência na prescrição
  - Scenario: Bedrock não identifica inconsistências
  - Scenario: Falha na chamada ao Bedrock

No real AWS call is made in this file — every boto3 Bedrock Runtime client
is a ``unittest.mock.MagicMock`` injected via the ``client`` parameter. The
one real Bedrock call (per the AWS validation budget) is run manually/
separately, not as part of the automated suite.
"""
import io
import json

import pandas as pd
import pytest
import streamlit as st
from botocore.exceptions import BotoCoreError, ClientError

from prescriptions.bedrock_review import (
    REQUIRED_COLUMNS,
    PrescriptionReviewError,
    PrescriptionValidationError,
    build_review_prompt,
    generate_alerts_for_findings,
    load_prescriptions,
    parse_bedrock_response,
    review_patient_prescriptions,
)


@pytest.fixture(autouse=True)
def clean_session_state():
    st.session_state.clear()
    yield
    st.session_state.clear()


# ── load_prescriptions (CSV/Excel loading + validation) ─────────────


def test_load_prescriptions_accepts_valid_csv():
    csv_bytes = (
        "paciente,medicamento,dose,data\n"
        "Paciente A,Losartana,50mg,2026-01-05\n"
        "Paciente A,Losartana,50mg,2026-01-19\n"
    ).encode("utf-8")

    df = load_prescriptions(io.BytesIO(csv_bytes), filename="prescricoes.csv")

    assert list(df.columns) == REQUIRED_COLUMNS
    assert len(df) == 2
    assert df.iloc[0]["paciente"] == "Paciente A"


def test_load_prescriptions_accepts_valid_excel():
    df_in = pd.DataFrame(
        {
            "paciente": ["Paciente B", "Paciente B"],
            "medicamento": ["Metformina", "Metformina"],
            "dose": ["500mg", "2000mg"],
            "data": ["2026-01-03", "2026-01-24"],
        }
    )
    buffer = io.BytesIO()
    df_in.to_excel(buffer, index=False)
    buffer.seek(0)

    df = load_prescriptions(buffer, filename="prescricoes.xlsx")

    assert list(df.columns) == REQUIRED_COLUMNS
    assert len(df) == 2


def test_load_prescriptions_is_case_insensitive_on_column_names():
    csv_bytes = (
        "Paciente,Medicamento,Dose,Data\n" "Paciente A,Losartana,50mg,2026-01-05\n"
    ).encode("utf-8")

    df = load_prescriptions(io.BytesIO(csv_bytes), filename="prescricoes.csv")

    assert list(df.columns) == REQUIRED_COLUMNS


def test_load_prescriptions_rejects_file_missing_required_column():
    csv_bytes = (
        "paciente,medicamento,data\n" "Paciente A,Losartana,2026-01-05\n"
    ).encode("utf-8")

    with pytest.raises(PrescriptionValidationError, match="dose"):
        load_prescriptions(io.BytesIO(csv_bytes), filename="prescricoes.csv")


def test_load_prescriptions_rejects_unparseable_file():
    with pytest.raises(PrescriptionValidationError):
        load_prescriptions(io.BytesIO(b"\x00\x01not a csv or excel file"), filename="prescricoes.csv")


# ── build_review_prompt ───────────────────────────────────────────────


def test_build_review_prompt_includes_patient_name_and_rows():
    df = pd.DataFrame(
        {
            "paciente": ["Paciente B", "Paciente B"],
            "medicamento": ["Metformina", "Metformina"],
            "dose": ["500mg", "2000mg"],
            "data": ["2026-01-03", "2026-01-24"],
        }
    )

    prompt = build_review_prompt(df, "Paciente B")

    assert "Paciente B" in prompt
    assert "Metformina" in prompt
    assert "500mg" in prompt
    assert "2000mg" in prompt
    assert "JSON" in prompt


# ── parse_bedrock_response ───────────────────────────────────────────


def test_parse_bedrock_response_parses_plain_json_array():
    raw = json.dumps(
        [
            {"tipo": "mudanca_dose_abrupta", "explicacao": "Dose subiu de 500mg para 2000mg sem titulação gradual."}
        ]
    )

    findings = parse_bedrock_response(raw)

    assert len(findings) == 1
    assert findings[0]["tipo"] == "mudanca_dose_abrupta"


def test_parse_bedrock_response_parses_json_embedded_in_surrounding_text():
    raw = (
        "Aqui está minha análise:\n"
        '[{"tipo": "interacao_medicamentosa", "explicacao": "Warfarina e Aspirina juntas aumentam risco de sangramento."}]'
        "\nEspero que ajude."
    )

    findings = parse_bedrock_response(raw)

    assert len(findings) == 1
    assert findings[0]["tipo"] == "interacao_medicamentosa"


def test_parse_bedrock_response_returns_empty_list_for_empty_array():
    findings = parse_bedrock_response("[]")

    assert findings == []


def test_parse_bedrock_response_raises_on_invalid_json():
    with pytest.raises(PrescriptionReviewError):
        parse_bedrock_response("isto não é JSON de forma alguma")


# ── generate_alerts_for_findings ─────────────────────────────────────


def test_generate_alerts_for_findings_creates_one_alert_per_finding():
    findings = [
        {"tipo": "mudanca_dose_abrupta", "explicacao": "Dose dobrou sem justificativa."},
        {"tipo": "interacao_medicamentosa", "explicacao": "Risco de sangramento."},
    ]

    alerts = generate_alerts_for_findings(findings, "Paciente B")

    assert len(alerts) == 2
    assert all(a.origin == "Prescrições" for a in alerts)
    assert "Paciente B" in alerts[0].description
    assert "Dose dobrou" in alerts[0].description


def test_generate_alerts_for_findings_returns_empty_list_when_no_findings():
    alerts = generate_alerts_for_findings([], "Paciente A")

    assert alerts == []


def test_generate_alerts_for_findings_sets_structured_category_and_level():
    # Structured fields (change alertas-estruturados-e-vinculo-sinais-vitais):
    # each inconsistency alert carries a "Inconsistência de prescrição"
    # category and the finding's ``tipo`` as level; the description text the
    # user sees is unchanged.
    findings = [
        {"tipo": "mudanca_dose_abrupta", "explicacao": "Dose dobrou sem justificativa."},
    ]

    alerts = generate_alerts_for_findings(findings, "Paciente B")

    assert len(alerts) == 1
    assert alerts[0].category == "Inconsistência de prescrição"
    assert alerts[0].level == "mudanca_dose_abrupta"
    assert "Dose dobrou" in alerts[0].description


def test_generate_alerts_for_findings_assigns_sequential_ids():
    # alert_id (change alertas-estruturados-audio-prescricoes): one unique
    # #P<NN> id per finding, in the same order as the findings list, so the
    # UI card and the feed alert can be matched.
    findings = [
        {"tipo": "mudanca_dose_abrupta", "explicacao": "Dose dobrou sem justificativa."},
        {"tipo": "interacao_medicamentosa", "explicacao": "Risco de sangramento."},
    ]

    alerts = generate_alerts_for_findings(findings, "Paciente B")

    assert [a.alert_id for a in alerts] == ["#P01", "#P02"]


# ── review_patient_prescriptions (mocked Bedrock client) ────────────


def _converse_response(text: str):
    return {"output": {"message": {"content": [{"text": text}]}}}


def test_review_patient_prescriptions_returns_findings_and_alerts():
    from unittest.mock import MagicMock

    client = MagicMock()
    client.converse.return_value = _converse_response(
        json.dumps(
            [
                {
                    "tipo": "mudanca_dose_abrupta",
                    "explicacao": "Dose de Metformina subiu de 500mg para 2000mg em uma semana.",
                }
            ]
        )
    )

    df = pd.DataFrame(
        {
            "paciente": ["Paciente B", "Paciente B"],
            "medicamento": ["Metformina", "Metformina"],
            "dose": ["500mg", "2000mg"],
            "data": ["2026-01-03", "2026-01-24"],
        }
    )

    result = review_patient_prescriptions(df, "Paciente B", client=client)

    assert len(result["findings"]) == 1
    assert len(result["alerts"]) == 1
    client.converse.assert_called_once()
    _, kwargs = client.converse.call_args
    assert "modelId" in kwargs


def test_review_patient_prescriptions_returns_no_findings_and_no_alerts_when_clean():
    from unittest.mock import MagicMock

    client = MagicMock()
    client.converse.return_value = _converse_response("[]")

    df = pd.DataFrame(
        {
            "paciente": ["Paciente A"],
            "medicamento": ["Losartana"],
            "dose": ["50mg"],
            "data": ["2026-01-05"],
        }
    )

    result = review_patient_prescriptions(df, "Paciente A", client=client)

    assert result["findings"] == []
    assert result["alerts"] == []


def test_review_patient_prescriptions_raises_clean_error_on_client_error():
    from unittest.mock import MagicMock

    client = MagicMock()
    client.converse.side_effect = ClientError(
        {"Error": {"Code": "AccessDeniedException", "Message": "not authorized"}}, "Converse"
    )

    df = pd.DataFrame(
        {"paciente": ["Paciente A"], "medicamento": ["Losartana"], "dose": ["50mg"], "data": ["2026-01-05"]}
    )

    with pytest.raises(PrescriptionReviewError):
        review_patient_prescriptions(df, "Paciente A", client=client)


def test_review_patient_prescriptions_raises_clean_error_on_timeout():
    from unittest.mock import MagicMock

    client = MagicMock()

    class _FakeTimeout(BotoCoreError):
        fmt = "fake timeout"

    client.converse.side_effect = _FakeTimeout()

    df = pd.DataFrame(
        {"paciente": ["Paciente A"], "medicamento": ["Losartana"], "dose": ["50mg"], "data": ["2026-01-05"]}
    )

    with pytest.raises(PrescriptionReviewError):
        review_patient_prescriptions(df, "Paciente A", client=client)


def test_review_patient_prescriptions_skips_leading_reasoning_content_block():
    """Some Claude models (extended thinking) prepend a reasoningContent
    block with no "text" key before the actual answer block; parsing
    must find the first block that does have "text" rather than assume
    index 0."""
    from unittest.mock import MagicMock

    client = MagicMock()
    client.converse.return_value = {
        "output": {
            "message": {
                "content": [
                    {"reasoningContent": {"reasoningText": {"text": "", "signature": "abc"}}},
                    {
                        "text": json.dumps(
                            [{"tipo": "mudanca_dose_abrupta", "explicacao": "Dose quadruplicou."}]
                        )
                    },
                ]
            }
        }
    }

    df = pd.DataFrame(
        {"paciente": ["Paciente A"], "medicamento": ["Losartana"], "dose": ["50mg"], "data": ["2026-01-05"]}
    )

    result = review_patient_prescriptions(df, "Paciente A", client=client)

    assert len(result["findings"]) == 1
    assert result["findings"][0]["tipo"] == "mudanca_dose_abrupta"


def test_review_patient_prescriptions_raises_clean_error_on_unparseable_response():
    from unittest.mock import MagicMock

    client = MagicMock()
    client.converse.return_value = _converse_response("resposta sem JSON nenhum")

    df = pd.DataFrame(
        {"paciente": ["Paciente A"], "medicamento": ["Losartana"], "dose": ["50mg"], "data": ["2026-01-05"]}
    )

    with pytest.raises(PrescriptionReviewError):
        review_patient_prescriptions(df, "Paciente A", client=client)
