# Code Generation Plan — u3-async-crm

> Unidade U3 (Async CRM). Escopo: `feature` · Estratégia: `standard` · Metodologia: `test-after` (Testing Contract do time). Greenfield.
> Fontes: contract-summary.md (Contratos 4 e 5), unit-of-work.md, requirements.md (FR11/FR7.3/NFR4.1/NFR5.1/NFR2.1), referência de convenções: u1-core-conversation e u2-async-voice.

## Testing Contract

```json
{
  "version": 1,
  "methodology": "test-after",
  "source": "team",
  "ordering": "Implementamos cada camada testável aplicável e, em seguida, escrevemos e executamos os testes dessa camada antes de avançar para a próxima.",
  "scope": "feature",
  "test_strategy": "standard",
  "project_type": "greenfield",
  "contract_sha256": "sha256:f1f75c7c9423814cb8f14b07dbb425fb0ba3324465c44388f61d1088141f6048"
}
```

Obrigações: 5-8 testes por componente; testes de unidade + integração para boundaries chave; piso de cobertura de 80% de linhas; execução em CI antes do merge. Nenhuma meta pode ser relaxada para fazer uma etapa passar.

## Steps

- [x] **Step 1** — Estrutura do projeto e configuração de produção (layout `apps/crm-adapter/{handler.py, service/, infra/, tests/, requirements.txt}` conforme PRD §7.4). *(História: setup da unidade U3)*
- [x] **Step 2** — Bootstrap do runner de testes (pytest + pytest-cov já no `.venv` da raiz; path bootstrap via `tests/conftest.py`, pyproject da raiz intocado) e registro do comando exato com escopo da unidade. *(Testing Contract: runner antes do primeiro teste)*
- [x] **Step 3** — Camada de dados: `infra/csv_store.py` CsvStore (workspace CSV do CRM simulado da POC) e `infra/session_store.py` SessionLookup (leitor do Contract 5, GSI `session-index`). *(FR11.2, Contract 5)*
- [x] **Step 4** — Testes das camadas de dados (test-after). *(Testing Contract)*
- [x] **Step 5** — Gateways e esteira: `service/crm_gateway.py` (`CrmGateway` Protocol + `CsvCrmGateway` CRM simulado default + `McpCrmGateway` via MCP com cliente injetado e import protegido do SDK), `service/flow_gateway.py` (`FlowGateway` Protocol + `HttpFlowGateway` callback de status) e `service/status_sync.py` (`KANBAN_STAGES`, regra `stage_for_lead`, `StatusSync` sincroniza esteira e devolve status ao fluxo). *(FR11.1, FR11.2, FR11.3, FR7.3)*
- [x] **Step 6** — Testes dos gateways e da esteira (test-after). *(Testing Contract)*
- [x] **Step 7** — Orquestrador + API: `service/crm_adapter.py` CrmAdapter (valida Contract 4 → lead desconhecido → upsert CRM → esteira Kanban → callback de status; outcomes `ok`/`retry`/`drop`; guarda de política de tentativas; logs estruturados PII-safe) e `handler.py` SQS batch handler (`batchItemFailures`, DLQ-friendly, nunca derruba o lote). *(Contract 4, FR11.3, NFR4.1, NFR5.1, NFR2.1)*
- [x] **Step 8** — Testes do orquestrador + integração (pipeline SQS end-to-end contra CSV real, lote parcial com drop/retry, política de DLQ, wiring do `handler()`). *(Testing Contract)*
- [x] **Step 9** — Configuração de build/deploy (`requirements.txt`) e documentação/traceability (`source-manifest.json`, `traceability.json`). *(Contract 4, Contract 5)*

## Rastreabilidade passo → requisito

| Step | Requisito(s) |
|------|--------------|
| 1 | — (estrutura) |
| 2 | Testing Contract |
| 3 | FR11.2, Contract 5 |
| 4 | Testing Contract |
| 5 | FR11.1, FR11.2, FR11.3, FR7.3 |
| 6 | Testing Contract |
| 7 | Contract 4, FR11.3, NFR4.1, NFR5.1, NFR2.1 |
| 8 | Testing Contract |
| 9 | Contract 4, Contract 5 |
