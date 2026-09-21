# Integration Test Instructions — Agente SDR Imobiliário (POC)

Estratégia de teste: **Standard** (test_strategy dos contratos de code-generation). Testes de unidade por unit já cobertos pelo estágio Code Generation; este documento cobre os **testes de integração entre units** e boundaries de serviço.

## Setup e configuração
- Framework: **pytest** (`.venv/bin/pytest`), sem fixture de rede — dependências externas são fakes/injeção.
- Cada suíte integração fica em `apps/<app>/tests/integration/`.
- `COVERAGE_FILE=/tmp/.cov-<unit>` para isolar cobertura.

## O que cada teste de integração cobre (boundary principal por unit)
| Unit | Suíte de integração | Boundary provado |
|---|---|---|
| u1 conversation-router | `apps/conversation-router/tests/integration/test_conversation_router.py` | Fluxo ponta-a-ponta: consent → intenção → qualificação → imóveis (RAG injetável) → agendamento → handoff; contrato com checker de restrição da U5 |
| u2 voice-adapter | `apps/voice-adapter/tests/integration/` | Handshake do handler: áudio (base64) → transcrição → evento normalizado para a U1 |
| u3 crm-adapter | `apps/crm-adapter/tests/integration/` | Sync de status de lead para CRM (CSV/fake HubSpot) + masking de PII na saída |
| u4 contact-ingest | `apps/contact-ingest/tests/integration/` | Ingest de contato → persistência de sessão com TTL → trigger de análise |
| u5 anomaly-detector | `apps/anomaly-detector/tests/integration/test_anomaly_pipeline.py` | Pipeline completo ingest→features→score→gate, incluindo contrato com a restrição consumida pela U1 |
| u6 followup | `apps/followup/tests/integration/` | Seleção de leads para follow-up (janela/cadência) → gravação de mensagens |
| u7 dashboard | `apps/dashboard-api/tests/integration/` | `GET /api/kpis` end-to-end (métricas agregadas a partir dos stores) |

## Como executar
```bash
# Todos os testes (unidade + integração) de uma unit — os de integração rodam junto:
COVERAGE_FILE=/tmp/.cov-u1 .venv/bin/python -m pytest apps/conversation-router/tests \
  --cov=apps/conversation-router --cov-report=term --cov-fail-under=80 -q

# Apenas integração de uma unit:
.venv/bin/python -m pytest apps/anomaly-detector/tests/integration -q
```

## Metas
- Cobertura ≥ 80% por unit (piso do contrato; atual ≥ 95.29% em todas).
- 0 falhas nos boundaries listados acima; qualquer falha em integration é bloqueante do estágio.

## Gestão de dados de teste
- Dados sintéticos nos fixtures (leads fake, CSVs temporários via `tmp_path`); nenhum dado real de cliente é usado.

## Testes não executáveis localmente (deferidos)
- Integração real com Telegram API, OpenRouter (LLM), DynamoDB/SQS/EventBridge e HubSpot via MCP: requerem ambiente provisionado — **owner: deployment-pipeline / deployment-execution / performance-validation** (ver `cross-unit-traceability.md`).
