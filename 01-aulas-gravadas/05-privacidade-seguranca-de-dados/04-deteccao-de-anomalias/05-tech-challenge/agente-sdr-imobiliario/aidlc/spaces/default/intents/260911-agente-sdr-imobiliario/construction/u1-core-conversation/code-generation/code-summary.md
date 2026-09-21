# Code Summary — u1-core-conversation

> Estágio Code Generation (Construction). Escopo `feature`, estratégia `standard`, metodologia `test-after`.

## Files created/modified

**Aplicação (`apps/conversation-router/`, estrutura do PRD §7.4):**
- `handler.py` — handler Lambda webhook Telegram (200/401/400), enfileira SQS voice/CRM, envia resposta Telegram
- `service/entities.py` — entidades Lead/Conversation (entities.md), TTL 90 dias
- `service/security_layer.py` — R1 (PII masking regex, persistência, detecção de vazamento, unmask), R6 (mensagem de consentimento LGPD), R7 (guardrails: prompt injection + tópicos negados)
- `service/flow/lead_qualifier.py` — R3 (score ponderado prontidão/urgência/ticket, explainability, threshold 70), FR10 (roleta: ≤500 m² rodízio, >500 especialista)
- `service/flow/sales_flow.py` — máquina de estados SalesFlow (greeting→elicitation→intent→qualification→recommendation→scheduling→handoff/followup), R2 (confiança ≥0.85), R4 (top-3 RAG), R5, R6
- `infra/session_store.py` — DynamoDB: get_or_create por telegram_user_id, save Lead+Conversation

**Testes:** `apps/conversation-router/tests/unit/test_entities.py` (8), `test_session_store.py` (4), `test_pii_masker.py` (8), `test_guardrails.py` (5), `test_lead_qualifier.py` (8), `test_sales_flow.py` (8), `tests/integration/test_conversation_router.py` (9)

**Config:** `pyproject.toml` (raiz), `apps/conversation-router/requirements.txt`, `requirements-dev.txt` (raiz)

## Key implementation decisions

- Handler recebe dependências por injeção (store, security, flow, sqs, telegram, secret) — testável sem AWS real; `handler()` de produção monta com boto3/env.
- PII mascarado antes do flow (R1/NFR2.1); saída do agente validada contra vazamento com fallback.
- SQS voice emite schema do Contract 3; SQS CRM emite schema do Contract 4.
- Validação do secret via header `X-Telegram-Bot-Api-Secret-Token` (Contract 1: 401/400/200).
- Score: prontidão+decisor 40, urgência (prazo ≤3 meses) 30, ticket (budget ≥50k) 30; não qualificado < 70 → Followup.

## Test coverage summary

- 50 testes (43 unit/integration… - total 8, store 4, pii 8, guardrails 5, qualifier 8, flow 8, router 9) — todos verdes.
- Cobertura `apps/conversation-router`: **92.16%** (piso 80% ✓). `security_layer.py`, `entities.py` 100%.
- Comando unit-scoped em `unit-test-instructions.md`.

## Deviations from the plan

- SalesFlow implementado como state machine própria com os mesmos estados/contratos da spec; LangGraph listado em `requirements.txt` para adoção no Build and Test (geração inicial evitou dependência pesada não necessária para a lógica testada).
- Persistência criptografada KMS da PII: `SecurityLayer` delega a um `pii_store` injetável; provisionamento KMS fica na infraestrutura (fora dos artefatos desta unidade).
- IaC (Terraform) não gerado neste estágio — handled pelos estágios de infrastructure/deployment.
- **Reorganização física do código** (decisão do humano, registrada neste estágio): layout inicial `src/sdr/{flow,handlers,security,storage}` realinhado à estrutura do PRD §7.4 — `apps/conversation-router/{handler.py, service/, infra/, tests/, requirements.txt}` (código-fonte → `service/`, adaptadores de dados → `infra/`, Lambda handler → `handler.py`). Imports absolutos reescritos (`service.*`, `infra.*`, `handler`); `pyproject.toml` da raiz ajustado (pythonpath/testpaths/cobertura → `apps/conversation-router`). Testes re-executados pós-reorg: 50 verdes, cobertura 92.16%.