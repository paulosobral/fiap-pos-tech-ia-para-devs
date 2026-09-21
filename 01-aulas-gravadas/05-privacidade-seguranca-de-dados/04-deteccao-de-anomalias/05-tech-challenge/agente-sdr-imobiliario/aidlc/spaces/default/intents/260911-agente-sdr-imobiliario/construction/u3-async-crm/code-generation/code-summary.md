# Code Summary — u3-async-crm

> Estágio Code Generation (Construction). Escopo `feature`, estratégia `standard`, metodologia `test-after`.
> Revisão it.2 (fix ROUND): consumidor alinhado à **Superfície de contrato interno (real)** da u1-core-conversation (seção do code-summary da u1).

## Files created/modified

**Aplicação (`apps/crm-adapter/`, estrutura do PRD §7.4):**
- `handler.py` — handler Lambda SQS (resposta parcial `batchItemFailures`, nunca derruba o lote); wiring de produção com boto3/env e DI por construtor; `INTERNAL_SECRET_TOKEN` **obrigatória** via `_env` (fail-fast), como `FLOW_BASE_URL`
- `service/crm_adapter.py` — orquestrador CrmAdapter: valida o payload REAL do Contract 4 → lead desconhecido (Contract 5) → upsert no CRM (com budget/deadline/area) → esteira Kanban (monotônica) → callback de status; outcomes `ok`/`retry`/`drop`; guarda de política de tentativas (`ApproximateReceiveCount`); logs estruturados JSON PII-safe no INFO **e nos erros** (NFR5.1 + NFR2.1)
- `service/crm_gateway.py` — interface `CrmGateway` (Protocol) + `CsvCrmGateway` (CRM simulado default da POC sobre workspace CSV, colunas de handoff budget/deadline/area) + `McpCrmGateway` (CRM real via MCP, cliente injetado, import do SDK protegido) + `build_hubspot_client` (FR11.1, wiring da demo ao vivo — ver Deviations)
- `service/status_sync.py` — `KANBAN_STAGES` (MESMA ordem monotônica da u1: novo → qualificado → contato-feito → visita-agendada → handoff → ganho/perdido), regra determinística `stage_for_lead` (score ≥ 70 ou urgência `high` → `qualificado`) e `StatusSync` monotônico (nunca regride estágio avançado no CRM) devolvendo o estágio aplicado ao fluxo
- `service/flow_gateway.py` — interface `FlowGateway` (Protocol) + `HttpFlowGateway` (callback de status no receptor REAL da u1: `POST /internal/crm-status`, header `X-Internal-Secret` obrigatório, body `{lead_id, session_id, stage}`)
- `service/pii_mask.py` (NOVO, rodada de fix) — `mask_pii`: padrões OFICIAIS de masking copiados do `SecurityLayer` da u1 (NOME/EMAIL/TELEFONE/CNPJ + lista controlada de nomes); uso exclusivo para logs (fronteira Lambda proíbe import cross-app)
- `infra/csv_store.py` — CsvStore (leitura/escrita atômica do CSV, valida header e colunas obrigatórias, cria diretórios); colunas + `budget`, `deadline`, `area` (handoff do Contract 4 atualizado)
- `infra/session_store.py` — SessionLookup (leitor do Contract 5 espelhando o acesso REAL do dono da tabela: get_item composto `LEAD#<lead_id>/PROFILE` + `LEAD#<lead_id>/CONV#<session_id>`; sem GSI inventado)

**Testes:** `tests/unit/test_crm_adapter.py` (31), `test_status_sync.py` (21), `test_csv_crm_gateway.py` (11), `test_csv_store.py` (11), `test_mcp_crm_gateway.py` (8), `test_flow_gateway.py` (8), `test_session_store.py` (5), `test_pii_mask.py` (7, NOVO), `tests/integration/test_crm_pipeline.py` (15)

**Config:** `apps/crm-adapter/requirements.txt`, `tests/conftest.py` (bootstrap de path). `pyproject.toml` da raiz, `apps/conversation-router/**` e `apps/voice-adapter/**` intocados; nenhuma outra config de raiz modificada.

## Key implementation decisions

- **Alinhamento ao payload real do produtor (Contract 4 atualizado pela u1):** `lead_data = {name, email, phone, score, urgency, intent, budget, deadline, area}` — `name` (sempre string não-vazia: `contact["NOME"]` com fallback seguro) e `urgency` (rótulos oficiais `low/medium/high`) obrigatórios; `email`/`phone`/`score`/`intent`/`budget`/`deadline`/`area` podem vir ausentes ou null. Mensagem sem email/phone é REGISTRADA com campos vazios e flag `contact_info_partial` no log — nunca descartada (FR11.3 executa no wiring real).
- DI por construtor em todos os componentes (crm, status, flow, sessions) — a suíte roda sem AWS real, sem SDK MCP e sem rede.
- Semântica SQS partial-batch (NFR4.1): `retry` → `batchItemFailures` (redrive SQS → DLQ, Contract 4); `drop` explícito para schema inválido, lead desconhecido (Contract 5) e corpo não-JSON; exceção inesperada → `retry` (o lote nunca quebra).
- Política de DLQ em duas camadas: o redrive do SQS (`maxReceiveCount` → DLQ) é o mecanismo primário; adicionalmente o orquestrador lê `ApproximateReceiveCount` do record e converte `retry` em `drop` quando a política configurável (`CRM_MAX_RECEIVES`, padrão 3) está esgotada.
- Desacoplado do fluxo do agente: callback de status via `FlowGateway` injetado — nenhum import de internals de `apps/conversation-router`; o shape segue o receptor real (`/internal/crm-status` da u1) e alimenta FR7.3.
- CRM atrás de interface: `CsvCrmGateway` (default POC) e `McpCrmGateway` trocáveis sem tocar no orquestrador; o import do SDK MCP é protegido no nível de módulo.
- PII-safe em profundidade (NFR2.1): `lead_data` nunca entra em logs; o preview do body SQS malformado (que ainda carrega PII) é MASCARADO com os padrões oficiais da u1 (`service/pii_mask.py`, cópia por fronteira Lambda) e truncado antes do CloudWatch.
- Logs estruturados JSON (`log_event`, com `level`) no INFO **e em todos os caminhos de erro** do orquestrador e dos gateways (NFR5.1) — testados via `caplog` parseando o JSON.
- Esteira Kanban monotônica: `StatusSync.sync` lê o estágio corrente do CRM (`get_lead`) e só escreve o estágio alvo quando a ordem `KANBAN_STAGES` permite (índice alvo ≥ corrente); estágio avançado — inclusive movido pelo corretor — é mantido e o estágio APLICADO é devolvido ao fluxo (espelho exato do receptor `/internal/crm-status` da u1, sem depender da validação dele).
- Estágio Kanban determinístico: score ≥ 70 ou urgência `high` → `qualificado`; caso contrário `novo`. Estágios posteriores (contato-feito em diante) são definidos externamente (corretor/CRM) e preservados monotonicamente.

## Test coverage summary

- 117 testes (102 unit + 15 integração) — todos verdes.
- Cobertura `apps/crm-adapter`: **99.59%** (piso 80% ✓). Misses restantes são ramos defensivos dentro de arquivos de teste.
- Comando unit-scoped registrado em `unit-test-instructions.md` (com `COVERAGE_FILE` isolado).

## Deviations from the plan

- CRM simulado implementado como **workspace CSV local** (FR11.2 "CSV/Excel"): `CsvCrmGateway` + `CsvStore` com escrita atômica (`mkstemp` + `os.replace`). Excel (openpyxl) não foi adicionado — CSV cobre a POC e evita dependência extra; a troca por Excel é um `CsvStore` alternativo atrás da mesma interface.
- Caminho do CRM simulado configurável via `CRM_CSV_PATH` (default `/tmp/crm-leads.csv` no Lambda — único diretório gravável); a semântica de persistência real (S3/EFS ou CRM verdadeiro) fica para Build and Test/infra.
- Política de DLQ adicional (`ApproximateReceiveCount` ≥ `CRM_MAX_RECEIVES` → `drop`) não estava especificada como mecanismo explícito; incluída como guarda contra redrive infinito quando a fila não tem DLQ configurada. O mecanismo primário continua sendo o redrive SQS → DLQ do Contract 4.
- `McpCrmGateway` opera sobre um `McpClient` Protocol (`call_tool(name, arguments)` síncrono) com ferramentas `crm_upsert_lead`/`crm_get_lead`/`crm_update_stage`; o SDK MCP real é async (transporte + handshake `initialize`) e a ponte é detalhe da demo (ver Deviations da rodada de fix) — o SDK nunca é exigido pela suíte.
- Header do CSV do CRM simulado estendido com `budget`/`deadline`/`area` (handoff do Contract 4 atualizado): arquivos com header antigo são rejeitados pelo `CsvStore` (header mismatch) — POC recria o arquivo; sem migração.
- IaC (Terraform) não gerado neste estágio — handled pelos estágios de infrastructure/deployment.

## Deviations da rodada de fix (iteração 2 do gate)

- **Contract 4 validado contra a forma real do produtor (R-01, Critical):** `name` obrigatório (a u1 corrigida sempre envia string não-vazia com fallback seguro — a corrupção semântica `name = decision_maker` foi corrigida no lado produtor); `email`/`phone` ausentes → registro com campos vazios + flag `contact_info_partial` no log `lead_synced` (sem descartar); `score` null aceitável; ids/timestamp continuam obrigatórios. Testes com snapshot REAL do `_enqueue_crm` da u1 (caso sem registro de PII e caso com decisor sem NOME).
- **Contract 5 lido pelo acesso real do dono (R-02, Critical):** `SessionLookup` reescrito — get_item composto `LEAD#<lead_id>/PROFILE` + `LEAD#<lead_id>/CONV#<session_id>` (uma leitura do CONV# confere lead e sessão ao mesmo tempo); GSI inexistente `session-index` removido; GSIs `telegram-user-index`/`lead-index` permanecem caminhos da u1. Teste fixa as chaves compostas exatas e a ausência de `query`.
- **Preview de body mascarado (R-03, Major):** `invalid_body` não loga mais o corpo cru — `mask_pii` (padrões oficiais copiados da u1 em `service/pii_mask.py`; import cross-app proibido pela fronteira Lambda) + truncagem a 120 chars + `body_length`. Teste de `caplog` prova ausência de PII e presença dos placeholders.
- **Callback alinhado ao receptor real (R-04, Major):** `POST /internal/crm-status` agora EXISTE na u1 (dono implementou) — `HttpFlowGateway` envia o shape exato (`{"lead_id", "session_id", "stage"}` como strings) com header `X-Internal-Secret` sempre presente; o secret passa a ser obrigatório no wiring (`_env`). Teste do shape.
- **Esteira monotônica (R-05, Major):** `KANBAN_STAGES` com a MESMA ordem oficial da u1 (`KANBAN_STAGE_ORDER`); `StatusSync.sync` nunca envia regressão — lê o estágio corrente no CRM e mantém o avançado (redelivery at-least-once/re-enfileiramento da u1 não derruba `contato-feito`/`visita-agendada` etc.). `urgency` usa os rótulos oficiais `low/medium/high` (`high` = urgente). Testes de regravação (unit + integração com CSV real) e de regressão bloqueada.
- **Handoff completo no CRM (R-06, Major):** `budget`/`deadline`/`area` do `lead_data` transferidos para a escrita do CRM — colunas novas no `CsvStore`/`CsvCrmGateway` e no `McpCrmGateway`; teste com payload completo (e2e CSV).
- **`INTERNAL_SECRET_TOKEN` obrigatória (R-07, Minor):** via `_env` (fail-fast), igual a `FLOW_BASE_URL`; teste de wiring sem a env levanta `RuntimeError`.
- **Logs estruturados nos erros (R-08, Minor):** `log_event` (JSON, com `level`) em todos os caminhos de falha de `crm_adapter` e dos gateways (`crm_unreachable`, `stage_sync_failed`, `flow_notify_failed`, `unexpected_failure`, `crm_store_*`, `mcp_tool_*`, `flow_unreachable`, `flow_rejected`); testes parseiam o JSON via `caplog`.
- **FR11.1 marcada honestamente (R-09, Minor):** `build_hubspot_client` devolve a CLASSE `ClientSession` com docstring que documenta o que falta para executar (bootstrap async de transporte + handshake `initialize` e adapter async→sync antes do `call_tool`); a demo ao vivo NÃO é executável sem conta HubSpot MCP — FR11.1 mudou de `OK` para `Deferred` em traceability.json (claim honesta; wiring da demo fica para o estágio de integração/demo).
