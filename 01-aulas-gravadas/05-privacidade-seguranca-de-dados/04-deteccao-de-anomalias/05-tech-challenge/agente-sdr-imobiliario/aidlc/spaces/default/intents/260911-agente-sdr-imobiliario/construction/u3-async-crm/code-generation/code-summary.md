# Code Summary — u3-async-crm

> Estágio Code Generation (Construction). Escopo `feature`, estratégia `standard`, metodologia `test-after`.

## Files created/modified

**Aplicação (`apps/crm-adapter/`, estrutura do PRD §7.4):**
- `handler.py` — handler Lambda SQS (resposta parcial `batchItemFailures`, nunca derruba o lote); wiring de produção com boto3/env e DI por construtor
- `service/crm_adapter.py` — orquestrador CrmAdapter: valida schema do Contract 4 → lead desconhecido (Contract 5) → upsert no CRM → esteira Kanban → callback de status; outcomes `ok`/`retry`/`drop`; guarda de política de tentativas (`ApproximateReceiveCount`); logs estruturados JSON PII-safe (NFR5.1 + NFR2.1)
- `service/crm_gateway.py` — interface `CrmGateway` (Protocol) + `CsvCrmGateway` (CRM simulado default da POC sobre workspace CSV) + `McpCrmGateway` (CRM real via MCP, cliente injetado, import do SDK protegido) + `build_hubspot_client` (FR11.1, demo ao vivo)
- `service/status_sync.py` — `KANBAN_STAGES` (esteira: novo → qualificado → contato-feito → visita-agendada → handoff → ganho/perdido), regra determinística `stage_for_lead` (score ≥ 70 ou urgência "alta" → `qualificado`) e `StatusSync` (grava estágio no CRM e devolve o status ao fluxo)
- `service/flow_gateway.py` — interface `FlowGateway` (Protocol) + `HttpFlowGateway` (callback de status via POST `/internal/crm-status` com `X-Internal-Secret`)
- `infra/csv_store.py` — CsvStore (leitura/escrita atômica do CSV, valida header e colunas obrigatórias, cria diretórios)
- `infra/session_store.py` — SessionLookup (leitor Contract 5, GSI `session-index`, valida `lead_id`)

**Testes:** `tests/unit/test_crm_adapter.py` (18), `test_status_sync.py` (13), `test_csv_crm_gateway.py` (11), `test_csv_store.py` (10), `test_mcp_crm_gateway.py` (8), `test_flow_gateway.py` (6), `test_session_store.py` (5), `tests/integration/test_crm_pipeline.py` (15)

**Config:** `apps/crm-adapter/requirements.txt`, `tests/conftest.py` (bootstrap de path). `pyproject.toml` da raiz, `apps/conversation-router/**` e `apps/voice-adapter/**` intocados; nenhuma outra config de raiz modificada.

## Key implementation decisions

- DI por construtor em todos os componentes (crm, status, flow, sessions) — a suíte roda sem AWS real, sem SDK MCP e sem rede.
- Semântica SQS partial-batch (NFR4.1): `retry` → `batchItemFailures` (redrive SQS → DLQ, Contract 4); `drop` explícito para schema inválido, lead desconhecido (sessão ausente no Contract 5) e corpo não-JSON; exceção inesperada → `retry` (o lote nunca quebra).
- Política de DLQ em duas camadas: o redrive do SQS (`maxReceiveCount` → DLQ) é o mecanismo primário; adicionalmente o orquestrador lê `ApproximateReceiveCount` do record e converte `retry` em `drop` quando a política configurável (`CRM_MAX_RECEIVES`, padrão 3) está esgotada — proteção contra redrive infinito em deploy sem DLQ.
- Desacoplado do fluxo do agente: callback de status via `FlowGateway` injetado — nenhum import de internals de `apps/conversation-router`; a esteira Kanban (`stage`) devolvida ao fluxo alimenta FR7.3.
- CRM atrás de interface: `CsvCrmGateway` (default POC) e `McpCrmGateway` trocáveis sem tocar no orquestrador; o import do SDK MCP é protegido no nível de módulo e `build_hubspot_client` existe para a demo única ao vivo (FR11.1), sem exigir o pacote na suíte.
- PII-safe: `lead_data` (nome/e-mail/telefone) nunca entra em logs — apenas ids, contagens e estágio (NFR2.1 em profundidade); PII sintética dos fixtures serve de guarda nos testes (`caplog`).
- Logs estruturados JSON (`log_event`) no mesmo estilo do u2 (NFR5.1).
- Estágio Kanban determinístico: score ≥ 70 ou urgência `alta` → `qualificado`; caso contrário `novo`. Regra da POC, substituível sem tocar no consumidor.

## Test coverage summary

- 86 testes (71 unit + 15 integração) — todos verdes.
- Cobertura `apps/crm-adapter`: **99.57%** (piso 80% ✓). Misses restantes são ramos defensivos dentro de arquivos de teste.
- Comando unit-scoped registrado em `unit-test-instructions.md`; suítes do u1 (50 testes, 92.16%) e u2 (58 testes, 99.86%) re-verificadas verdes após o trabalho.

## Deviations from the plan

- CRM simulado implementado como **workspace CSV local** (FR11.2 "CSV/Excel"): `CsvCrmGateway` + `CsvStore` com escrita atômica (`mkstemp` + `os.replace`). Excel (openpyxl) não foi adicionado — CSV cobre a POC e evita dependência extra; a troca por Excel é um `CsvStore` alternativo atrás da mesma interface.
- Caminho do CRM simulado configurável via `CRM_CSV_PATH` (default `/tmp/crm-leads.csv` no Lambda — único diretório gravável); a semântica de persistência real (S3/EFS ou CRM verdadeiro) fica para Build and Test/infra.
- Contrato HTTP de callback de status não existia no contract-summary (o Contract 4 cobre apenas o SQS de CRM): `HttpFlowGateway` publica `POST {FLOW_BASE_URL}/internal/crm-status` com header `X-Internal-Secret`, espelhando o padrão de re-injeção do u2. Endpoint presumido; ajuste final fica para Build and Test.
- Política de DLQ adicional (`ApproximateReceiveCount` ≥ `CRM_MAX_RECEIVES` → `drop`) não estava especificada como mecanismo explícito; foi incluída como guarda contra redrive infinito quando a fila não tem DLQ configurada. O mecanismo primário continua sendo o redrive SQS → DLQ do Contract 4.
- `McpCrmGateway` opera sobre um `McpClient` Protocol (`call_tool(name, arguments)`) com ferramentas `crm_upsert_lead`/`crm_get_lead`/`crm_update_stage`; o wiring real do MCP HubSpot (auth app, Inspector) é detalhe da demo ao vivo (FR11.1) e não do consumidor — o SDK nunca é exigido pela suíte.
- IaC (Terraform) não gerado neste estágio — handled pelos estágios de infrastructure/deployment.
