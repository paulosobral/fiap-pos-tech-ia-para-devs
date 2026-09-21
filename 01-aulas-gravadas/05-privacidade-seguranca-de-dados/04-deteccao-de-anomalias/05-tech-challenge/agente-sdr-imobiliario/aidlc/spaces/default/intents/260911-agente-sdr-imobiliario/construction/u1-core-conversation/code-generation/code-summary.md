# Code Summary — u1-core-conversation

> Estágio Code Generation (Construction). Escopo `feature`, estratégia `standard`, metodologia `test-after`.

## Files created/modified

**Aplicação (`apps/conversation-router/`, estrutura do PRD §7.4):**
- `handler.py` — handler Lambda webhook Telegram (200/401/400), enfileira SQS voice/CRM, envia resposta ao Telegram (cliente `TelegramApi` injetado na composição de produção; token via env/Secrets Manager); sincroniza `consent_recorded` do fluxo (sem auto-consent), persiste `route` do Lead e preenche `lead_data` do Contract 4 a partir do registro de PII
- `service/entities.py` — entidades Lead/Conversation (entities.md), TTL 90 dias; Lead inclui campo `route` (FR10.3)
- `service/security_layer.py` — R1 (PII masking regex — nomes completos e nome único da lista controlada, persistência, detecção de vazamento incl. NOME, unmask), R6 (mensagem de consentimento LGPD), R7 (guardrails: prompt injection + tópicos negados); `KmsPiiRegistry` (registro de PII criptografado via KMS em DynamoDB)
- `service/flow/lead_qualifier.py` — R3 (score ponderado prontidão/urgência/ticket, explainability, threshold 70; orçamento normalizado de unidades mil/k/milhão antes de comparar com 50.000), FR10 (roleta: ≤500 m² rodízio, >500 especialista)
- `service/flow/sales_flow.py` — máquina de estados SalesFlow (greeting→elicitation→intent→qualification→recommendation→scheduling→handoff/followup), R2 (confiança ≥0.85), R4 (top-3 RAG), R5, R6; extrator de estrutura do lead (`extract_lead_structure`: metragem, região, orçamento, prazo, nº de pessoas, decisor) sobre texto já mascarado, persistido em `context["lead_info"]`; consentimento registrado na decisão do lead (elicitation) e rota atribuída na qualificação
- `infra/session_store.py` — DynamoDB: get_or_create por telegram_user_id, save Lead+Conversation; leitura e escrita usam a mesma chave composta PK/SK (LEAD#…/PROFILE e LEAD#…/CONV#…)

**Testes:** `apps/conversation-router/tests/unit/test_entities.py` (9), `test_session_store.py` (5), `test_pii_masker.py` (11), `test_guardrails.py` (5), `test_lead_qualifier.py` (10), `test_sales_flow.py` (14), `tests/integration/fixtures.py`, `tests/integration/test_conversation_router.py` (14)

**Config:** `pyproject.toml` (raiz), `apps/conversation-router/requirements.txt`, `requirements-dev.txt` (raiz)

## Key implementation decisions

- Handler recebe dependências por injeção (store, security, flow, sqs, telegram, pii_store, secret) — testável sem AWS real; `handler()` de produção monta com boto3/env (`TELEGRAM_BOT_TOKEN`, `PII_KMS_KEY_ID`, `SPECIALIST_ROTATION` — valores injetados pelo deploy via Secrets Manager).
- PII mascarado antes do flow (R1/NFR2.1); saída do agente validada contra vazamento (EMAIL/TELEFONE/CNPJ/NOME) com fallback. Nome único só é mascarado se constar de lista controlada de nomes próprios pt-BR (`COMMON_FIRST_NAMES`) — limitação conhecida: nome fora da lista vaza ao LLM na entrada.
- Consentimento LGPD: `/start` não grava consentimento; `consent_recorded` só é gravado pela decisão do lead no fluxo (recusa "não" persiste `False`) e sincronizado da `flow_state` para a `Conversation` (R6/NFR2.3).
- Extrator de estrutura do lead roda sobre a mensagem já mascarada a cada turno e faz merge incremental em `conversation.context["lead_info"]` (FR2.3/FR3.1 sem estado pré-semeado).
- Na qualificação com score ≥ 70, `route()` é invocado (roleta/specialist) e a rota é persistida no Lead (FR10.3).
- SQS voice emite schema do Contract 3; SQS CRM emite schema do Contract 4 com `lead_data.email/phone` carregados do registro de PII no handoff.
- Validação do secret via header `X-Telegram-Bot-Api-Secret-Token` (Contract 1: 401/400/200).
- Score: prontidão+decisor 40, urgência (prazo ≤3 meses) 30, ticket (budget ≥50k) 30; não qualificado < 70 → Followup.

## Test coverage summary

- 68 testes (54 unit + 14 integração — unit: entities 9, store 5, pii 11, guardrails 5, qualifier 10, flow 14; integração: router 14) — todos verdes.
- Cobertura `apps/conversation-router`: **94.25%** (piso 80% ✓). `security_layer.py`, `entities.py` 100%.
- Comando unit-scoped em `unit-test-instructions.md`.

## Deviations from the plan

- **Redução de escopo intencional desta unidade (pontos de injeção com fallback):** SDR Agent (respostas humanizadas via OpenRouter), PropertiesRAG (FAISS/S3), Scheduler real e Handoff builder NÃO são implementados aqui — `SalesFlow` os recebe como callables injetáveis e segue com fallbacks determinísticos. As implementações reais pertencem às unidades/estágios posteriores (u2+).
- **Persistência criptografada KMS da PII:** `SecurityLayer`/`KmsPiiRegistry` usam um `pii_store` injetável (blob KMS em DynamoDB); o provisionamento da tabela DynamoDB e da chave KMS é o dono do schema — fica no estágio de IaC (fora dos artefatos desta unidade).
- **Schema de tabela DynamoDB (chave composta PK/SK: `LEAD#…/PROFILE`, `LEAD#…/CONV#…`, `PII#…/PII`):** definido aqui para o store; a tabela real (GSI `telegram-user-index`/`lead-index`, TTL) será provisionada pelo estágio de IaC, que é o dono final do schema.
- **Mascaramento de NOME:** nome único (token capitalizado isolado) só é mascarado quando consta de lista controlada de nomes próprios pt-BR; nomes completos (duas+ tokens capitalizados) são mascarados por regex. Nome fora da lista não é mascarado na entrada — ampliação (lista completa/detecção por NER) fica para a integração do SDR Agent.
- **Deps removidas de `requirements.txt`:** `langgraph` e `litellm` não são referenciados pelo código desta unidade (LLM entra como callable injetável; adoção real de OpenRouter/LangGraph acontece nas unidades posteriores) — removidas para não embarcar dependência sem uso.
- IaC (Terraform) não gerado neste estágio — handled pelos estágios de infrastructure/deployment.
- **Reorganização física do código** (decisão do humano, registrada neste estágio): layout inicial `src/sdr/{flow,handlers,security,storage}` realinhado à estrutura do PRD §7.4 — `apps/conversation-router/{handler.py, service/, infra/, tests/, requirements.txt}` (código-fonte → `service/`, adaptadores de dados → `infra/`, Lambda handler → `handler.py`). Imports absolutos reescritos (`service.*`, `infra.*`, `handler`); `pyproject.toml` da raiz ajustado (pythonpath/testpaths/cobertura → `apps/conversation-router`). Testes re-executados pós-reorg: 68 verdes, cobertura 94.25%.
