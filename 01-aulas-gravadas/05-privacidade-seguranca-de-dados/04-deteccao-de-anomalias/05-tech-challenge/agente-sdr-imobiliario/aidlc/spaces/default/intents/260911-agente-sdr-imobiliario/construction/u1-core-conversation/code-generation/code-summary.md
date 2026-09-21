# Code Summary — u1-core-conversation

> Estágio Code Generation (Construction). Escopo `feature`, estratégia `standard`, metodologia `test-after`.
> Revisão it.2 (fix ROUND): consuma a seção **Superfície de contrato interno (real)** abaixo — os consumidores (u2/u3/u4/u5) devem alinhar-se a ela.

## Files created/modified

**Aplicação (`apps/conversation-router/`, estrutura do PRD §7.4):**
- `handler.py` — handler Lambda com roteamento por path: webhook Telegram (200/401/400) + endpoints internos `POST /internal/inbound-text` e `POST /internal/crm-status` (autenticação `X-Internal-Secret`); enfileira SQS voice/CRM, envia resposta ao Telegram (cliente `TelegramApi` injetado na composição de produção; token via env/Secrets Manager); sincroniza `consent_recorded` do fluxo (sem auto-consent), persiste `route` e `urgency` do Lead e preenche `lead_data` do Contract 4 a partir do registro de PII
- `service/entities.py` — entidades Lead/Conversation (entities.md), TTL 90 dias; Lead inclui campos `route` (FR10.3); constantes `KANBAN_STAGE_ORDER` e `KANBAN_STATUS_MAP` (espelho da esteira do u3, FR7.3/FR11.3)
- `service/security_layer.py` — R1 (PII masking regex — nomes completos e nome único da lista controlada, persistência, detecção de vazamento incl. NOME, unmask), R6 (mensagem de consentimento LGPD), R7 (guardrails: prompt injection + tópicos negados); `KmsPiiRegistry` (registro de PII criptografado via KMS em DynamoDB)
- `service/flow/lead_qualifier.py` — R3 (score ponderado prontidão/urgência/ticket, explainability, threshold 70; orçamento normalizado de unidades mil/k/milhão/milhões antes de comparar com 50.000; prazo em semanas/dias normalizado p/ meses), `urgency()` (low/medium/high pelos MESMOS fatores do score), FR10 (roleta: ≤500 m² rodízio, >500 especialista)
- `service/flow/sales_flow.py` — máquina de estados SalesFlow (greeting→elicitation→intent→qualification→recommendation→scheduling→handoff/followup), R2 (confiança ≥0.85), R4 (top-3 RAG), R5, R6; extrator de estrutura do lead (`extract_lead_structure`: metragem, região, orçamento, prazo, nº de pessoas, decisor) sobre texto já mascarado, persistido em `context["lead_info"]`; `restriction_check` injetável consultado antes de decisões autônomas de agendamento/follow-up (FR9.4, fail-open); consentimento registrado na decisão do lead (elicitation) e rota atribuída na qualificação
- `service/restriction.py` (NOVO) — `DynamoRestrictionCheck`: checker de restrição de agendamento (FR9.4) que lê a tabela de alertas da U5 (env `ALERTS_TABLE`, default `sdr-alerts`, GSI `lead-index`; itens `scheduling_restricted=True` + `status="open"`), fail-open em erro
- `infra/session_store.py` — DynamoDB: get_or_create por telegram_user_id, save Lead+Conversation, `get_lead(lead_id)`, `save_lead(lead)`, `get_conversation(lead_id, session_id)`; leitura e escrita usam a mesma chave composta PK/SK (LEAD#…/PROFILE e LEAD#…/CONV#…) — esquema inalterado

**Testes:** `apps/conversation-router/tests/unit/test_entities.py` (9), `test_session_store.py` (10), `test_pii_masker.py` (11), `test_guardrails.py` (5), `test_lead_qualifier.py` (16), `test_sales_flow.py` (23), `test_restriction.py` (9, NOVO), `tests/integration/fixtures.py`, `tests/integration/test_conversation_router.py` (33)

**Config:** `pyproject.toml` (raiz), `apps/conversation-router/requirements.txt`, `requirements-dev.txt` (raiz)

## Superfície de contrato interno (real — referência para consumidores)

Todos os endpoints internos usam o MESMO mecanismo: header `X-Internal-Secret` comparado à env **`INTERNAL_SECRET_TOKEN`** (obrigatória em produção via `_env`; sem env configurada o endpoint responde **503**). Método estrito **POST** (outros → 405). Body JSON (não-JSON ou campos inválidos → 400).

| Endpoint | Body (exato) | Respostas | Semântica |
|---|---|---|---|
| `POST /internal/inbound-text` | `{"telegram_user_id": int, "session_id": str, "text": str}` (todos obrigatórios) | 200/400/401/405/503 | Recupera a sessão por `telegram_user_id` (GSI `telegram-user-index`); se a `session_id` enviada existir como linha `CONV#` do lead, é usada, senão cai para a sessão armazenada. Alimenta o fluxo como mensagem do lead (guard → máscara PII → SalesFlow → handoff CRM quando qualificado). Retorno 200: `{"ok": true, "lead_id", "session_id", "state", "response"}` |
| `POST /internal/crm-status` | `{"lead_id": str, "session_id": str, "stage": str}` (todos obrigatórios; `stage` ∈ `KANBAN_STAGE_ORDER`) | 200/400/401/404/405/503 | Atualiza o estágio do Lead de forma MONOTÔNICA: `("novo", "qualificado", "contato-feito", "visita-agendada", "handoff", "ganho", "perdido")` — regressão é ignorada (mantém o estágio já avançado; retorno informa o estágio aplicado). Lead inexistente → 404. `lead.status` "new"/"qualified" projetam-se em "novo"/"qualificado" para a comparação |

**Payload do CRM (Contract 4, SQS CRM — atualizado):** `lead_data = {name, email, phone, score, urgency, intent, budget, deadline, area}` — `email/phone` do registro de PII (KMS); `name` de `contact["NOME"]` (fallback: `"Decisor (nome não informado)"` se `decision_maker == "yes"`, senão `"Lead (nome não informado)"` — nunca a flag yes/no); `urgency` ∈ `low/medium/high` derivada pela MESMA regra de fatores do qualifier: **high = prazo curto (≤ 3 meses, com semanas/dias normalizados) E orçamento ≥ 50k; medium = exatamente um; low = nenhum**; `budget/deadline/area` = `context["lead_info"]` persistido (também gravados nos campos do Lead).

**Restrição de agendamento (FR9.4 — consumo da U5):** `SalesFlow(restriction_check=...)` consultado antes de agendamento autônomo (`_handle_scheduling`: restrito → scheduler NÃO é chamado, `scheduling_restricted=True`, decisão adiada para o corretor) e antes de follow-up autônomo (`_handle_followup`: restrito → `followup_deferred=True`, sem outreach). Em produção o checker é `DynamoRestrictionCheck` (tabela de alertas da U5 — env **`ALERTS_TABLE`**, default `sdr-alerts`, GSI `lead-index`), **fail-open** (erro = não restrito).

## Key implementation decisions

- Handler recebe dependências por injeção (store, security, flow, sqs, telegram, pii_store, secret, internal_secret) — testável sem AWS real; `handler()` de produção monta com boto3/env (`TELEGRAM_BOT_TOKEN`, `PII_KMS_KEY_ID`, `INTERNAL_SECRET_TOKEN`, `ALERTS_TABLE`, `SPECIALIST_ROTATION` — valores injetados pelo deploy via Secrets Manager; `INTERNAL_SECRET_TOKEN` é obrigatória via `_env`).
- PII mascarado antes do flow (R1/NFR2.1); saída do agente validada contra vazamento (EMAIL/TELEFONE/CNPJ/NOME) com fallback. Nome único só é mascarado se constar de lista controlada de nomes próprios pt-BR (`COMMON_FIRST_NAMES`) — limitação conhecida: nome fora da lista vaza ao LLM na entrada.
- Consentimento LGPD: `/start` não grava consentimento; `consent_recorded` só é gravado pela decisão do lead no fluxo (recusa "não" persiste `False`) e sincronizado da `flow_state` para a `Conversation` (R6/NFR2.3).
- Extrator de estrutura do lead roda sobre a mensagem já mascarada a cada turno e faz merge incremental em `conversation.context["lead_info"]` (FR2.3/FR3.1 sem estado pré-semeado).
- Na qualificação com score ≥ 70, `route()` é invocado (roleta/specialist) e a rota é persistida no Lead (FR10.3).
- SQS voice emite schema do Contract 3; SQS CRM emite schema do Contract 4 com `lead_data.email/phone/name/urgency/budget/deadline/area` (ver superfície de contrato acima).
- Validação do secret do webhook via header `X-Telegram-Bot-Api-Secret-Token` (Contract 1: 401/400/200); endpoints internos usam `X-Internal-Secret`.
- Score: prontidão+decisor 40, urgência (prazo ≤3 meses, semanas/dias normalizados) 30, ticket (budget ≥50k, escala milhão correta) 30; não qualificado < 70 → Followup.

## Test coverage summary

- 116 testes (83 unit + 33 integração — unit: entities 9, store 10, pii 11, guardrails 5, qualifier 16, flow 23, restriction 9; integração: router 33) — todos verdes.
- Cobertura `apps/conversation-router`: **96.51%** (piso 80% ✓).
- Comando unit-scoped em `unit-test-instructions.md`.

## Deviations from the plan

- **Redução de escopo intencional desta unidade (pontos de injeção com fallback):** SDR Agent (respostas humanizadas via OpenRouter), PropertiesRAG (FAISS/S3), Scheduler real e Handoff builder NÃO são implementados aqui — `SalesFlow` os recebe como callables injetáveis e segue com fallbacks determinísticos. As implementações reais pertencem às unidades/estágios posteriores (u2+).
- **Persistência criptografada KMS da PII:** `SecurityLayer`/`KmsPiiRegistry` usam um `pii_store` injetável (blob KMS em DynamoDB); o provisionamento da tabela DynamoDB e da chave KMS é o dono do schema — fica no estágio de IaC (fora dos artefatos desta unidade).
- **Schema de tabela DynamoDB (chave composta PK/SK: `LEAD#…/PROFILE`, `LEAD#…/CONV#…`, `PII#…/PII`):** definido aqui para o store; a tabela real (GSI `telegram-user-index`/`lead-index`, TTL) será provisionada pelo estágio de IaC, que é o dono final do schema.
- **Mascaramento de NOME:** nome único (token capitalizado isolado) só é mascarado quando consta de lista controlada de nomes próprios pt-BR; nomes completos (duas+ tokens capitalizados) são mascarados por regex. Nome fora da lista não é mascarado na entrada — ampliação (lista completa/detecção por NER) fica para a integração do SDR Agent.
- **Deps removidas de `requirements.txt`:** `langgraph` e `litellm` não são referenciados pelo código desta unidade (LLM entra como callable injetável; adoção real de OpenRouter/LangGraph acontece nas unidades posteriores) — removidas para não embarcar dependência sem uso.
- IaC (Terraform) não gerado neste estágio — handled pelos estágios de infrastructure/deployment.
- **Reorganização física do código** (decisão do humano, registrada neste estágio): layout inicial `src/sdr/{flow,handlers,security,storage}` realinhado à estrutura do PRD §7.4 — `apps/conversation-router/{handler.py, service/, infra/, tests/, requirements.txt}` (código-fonte → `service/`, adaptadores de dados → `infra/`, Lambda handler → `handler.py`). Imports absolutos reescritos (`service.*`, `infra.*`, `handler`); `pyproject.toml` da raiz ajustado (pythonpath/testpaths/cobertura → `apps/conversation-router`). Testes re-executados pós-reorg: 68 verdes, cobertura 94.25%.

## Deviations da rodada de fix (iteração 2 do gate)

- **FR2.3 marcado como Deferred no traceability.json** (R-02 it.2): a claim "diferenciar perfil B2B (principal) vs investidor PF" NÃO é materializada por código nesta unidade — o extrator captura metragem/região/orçamento/prazo/pessoas/decisor, mas não há classificação de perfil B2B/PF. A detecção de perfil fica para a integração do SDR Agent (u2+); nada foi implementado de forma artificial para a claim passar.
- **Urgência do handoff derivada e documentada** (R-05 it.2): `lead_data.name` agora vem de `contact["NOME"]` do registro de PII (fallback seguro, nunca a flag `decision_maker`); `urgency` reutiliza `LeadQualifier.urgency()` — regra documentada na seção "Superfície de contrato interno"; `budget/deadline/area` incluídos em `lead_data` (fechamento do gap apontado pelo R-06 da U3 — consumidores u3 ajustam suas colunas em sua própria rodada).
- **Endpoints internos implementados como dono** (R-01 da U2/U4, R-04 da U3): `/internal/inbound-text` e `/internal/crm-status` com `X-Internal-Secret` == env `INTERNAL_SECRET_TOKEN`; formas exatas na seção de contrato. Consumidores u2/u4 NÃO precisam mudar payload/header (o shape real do gateway deles foi honrado); a U3 ajustará monotonicidade do lado dela (a U1 já garante sem regressão).
- **`is_restricted(lead_id)` wired no fluxo** (R-02 da U5): `restriction_check` injetável em `SalesFlow`; em produção lê a MESMA tabela de alertas da U5 (env `ALERTS_TABLE`, default `sdr-alerts`, GSI `lead-index`, itens `scheduling_restricted=True` + `status="open"`), fail-open em erro. O Consumo real da U5 não muda — só a U1 passou a consultar.
- **Correção da extração de orçamento em escala milhão** (R-01 it.2): separadores de milhar múltiplos ("R$ 1.500.000" → 1.500.000) e singular "milhão" (o `Ã` de "milhão" não casava com `h[õo]es` nem com `ão` → caía em "mil", erro ×1000). Extração preserva o valor completo; `_budget_value` normaliza. E2E com ≥ R$ 1 mi e prazo de 6 meses → score 80 (qualifica).
- **Prazo em semanas/dias normalizado** (R-03 it.2): `_deadline_months` — "2 semanas" ≈ 0,47 mês, "15 dias" = 0,5 mês (base 30 dias/mês) para a regra R3.
- **Métodos aditivos no `SessionStore`** (`get_lead`, `save_lead`, `get_conversation`): sem mudança de esquema — PK/SK compostos LEAD#/CONV# permanecem exatamente como estavam (os consumidores espelharão o acesso real, não o contrário).
