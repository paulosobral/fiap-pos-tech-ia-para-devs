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
| u7 dashboard-api | `apps/dashboard-api/tests/integration/` + `tests/unit/test_leads.py` | `GET /api/kpis` end-to-end; `GET /api/leads` (contato decifrado do registro de PII) e `POST /api/leads/{id}/crm` (publica na fila do CRM, mesmo contrato do handoff) |
| u7 dashboard-ui | `apps/dashboard-ui/tests/unit/test_app_smoke.py` | Login/refresh Cognito, cofre de sessão, header, tabela de leads e botão HubSpot (streamlit `AppTest`) |
| u3 crm-adapter ↔ HubSpot | `apps/crm-adapter/tests/unit/test_hubspot_mcp.py` | `HubSpotCrmGateway`/`HubSpotMcpClient` com sessão MCP fake: busca por e-mail→telefone, create/update, rotação do refresh token |
| u1 → u5 → u7 (LLM real) | `apps/conversation-router/tests/quality/test_anomaly_e2e_quality.py` | Chat real grava a conversa → handler real do detector gera alerta + restringe agendamento → `/api/kpis` mostra o alerta sem `features` |
| infra ↔ código | `tests/infra/test_dynamodb_indexes.py` (roda no `start.sh`) | Todo índice DynamoDB consultado pelo código existe no Terraform da tabela certa (ADR-024; os DynamoDB falsos aceitam qualquer índice) |
| u1 (LLM real) | `apps/conversation-router/tests/quality/test_llm_quality_gate.py` | Naturalidade e fechamento de lead ponta a ponta com roteador e humanização reais |

## Como executar
```bash
# Todos os testes (unidade + integração) de uma unit — os de integração rodam junto:
COVERAGE_FILE=/tmp/.cov-u1 .venv/bin/python -m pytest apps/conversation-router/tests \
  --cov=apps/conversation-router --cov-report=term --cov-fail-under=80 -q

# Apenas integração de uma unit:
.venv/bin/python -m pytest apps/anomaly-detector/tests/integration -q
```

## Metas
- Cobertura ≥ 80% por unit (piso do contrato; atual: u1 89,60%, dashboard-ui 83,28%, demais ≥ 97,75%).
- Gate com LLM real: 59/59 (executado pelo `start.sh` antes do deploy).
- 0 falhas nos boundaries listados acima; qualquer falha em integration é bloqueante do estágio.

## Gestão de dados de teste
- Dados sintéticos nos fixtures (leads fake, CSVs temporários via `tmp_path`); nenhum dado real de cliente é usado.

## Testes não executáveis localmente (deferidos)
- Já exercitados com serviço real, **de forma manual** (2026-10-08): OpenRouter (gate de qualidade, automatizado), HubSpot via MCP (contato criado/atualizado, lead real do Telegram) e o stack na AWS (`start.sh`/`stop.sh`).
- Ainda sem teste automatizado contra a AWS real: webhook Telegram → API Gateway → ECS, DynamoDB/SQS/EventBridge, Cognito/JWT authorizer e o botão do dashboard no navegador — **owner: deployment-execution** (smoke do `start.sh` cobre parte).
