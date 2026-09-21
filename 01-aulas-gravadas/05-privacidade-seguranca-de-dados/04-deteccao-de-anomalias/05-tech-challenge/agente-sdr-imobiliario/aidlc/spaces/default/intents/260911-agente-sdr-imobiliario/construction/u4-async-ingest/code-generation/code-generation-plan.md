# Code Generation Plan — u4-async-ingest

> Unidade U4 (Async Ingest). Escopo: `feature` · Estratégia: `standard` · Metodologia: `test-after` (Testing Contract do time). Greenfield.
> Fontes: contract-summary.md (Contrato 5), unit-of-work.md, requirements.md (FR12/NFR4.1/NFR5.1/NFR2.1), referência de convenções: u1-core-conversation (SessionStore writer), u2-async-voice (re-injeção no router) e u3-async-crm.

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

- [x] **Step 1** — Estrutura do projeto e configuração de produção (layout `apps/contact-ingest/{handler.py, service/, infra/, tests/, requirements.txt}` conforme PRD §7.4). *(História: setup da unidade U4)*
- [x] **Step 2** — Bootstrap do runner de testes (pytest + pytest-cov já no `.venv` da raiz; path bootstrap via `tests/conftest.py`, pyproject da raiz intocado) e registro do comando exato com escopo da unidade. *(Testing Contract: runner antes do primeiro teste)*
- [x] **Step 3** — Camada de dados: `infra/dedupe_store.py` DedupeStore (put condicional por `message_id`, anti spam/duplicata) e `infra/session_store.py` SessionWriter (escritor do Contrato 5: Lead + Conversation, espelhando o padrão de escrita do SessionStore do u1). *(FR12, Contrato 5)*
- [x] **Step 4** — Testes das camadas de dados (test-after). *(Testing Contract)*
- [x] **Step 5** — Parsing e gateway: `service/email_parser.py` (Protocol `EmailParser` + `HeuristicEmailParser` heurístico para nome/e-mail/telefone, tolerante a formatos de portal, com corpo MIME cru opcional) e `service/router_gateway.py` (Protocol `RouterGateway` + `HttpRouterGateway` re-injeção `POST /internal/inbound-text` com `X-Internal-Secret`, espelhando o padrão do u2). *(FR12.2, FR12.3)*
- [x] **Step 6** — Testes do parser e do gateway (test-after). *(Testing Contract)*
- [x] **Step 7** — Orquestrador + handler: `service/contact_ingest.py` ContactIngest (parse → dedupe → abrir sessão → re-injetar 1ª mensagem no router; outcomes `ok`/`retry`/`drop`; rollback da marca de dedupe em falha transitória; logs estruturados PII-safe) e `handler.py` handler SES (wiring de produção; e-mail inválido → `drop` explícito, nunca derruba o evento; `PendingRetryError` sinaliza retry async da Lambda). *(FR12.1, FR12.3, NFR4.1, NFR5.1, NFR2.1)*
- [x] **Step 8** — Testes do orquestrador + integração (pipeline SES end-to-end com parser real, duplicatas no mesmo evento, formatos de portal variados, PII-safe nos logs, wiring do `handler()`). *(Testing Contract)*
- [x] **Step 9** — Configuração de build/deploy (`requirements.txt`) e documentação/traceability (`source-manifest.json`, `traceability.json`). *(Contrato 5, FR12)*

## Rastreabilidade passo → requisito

| Step | Requisito(s) |
|------|--------------|
| 1 | — (estrutura) |
| 2 | Testing Contract |
| 3 | FR12, Contrato 5 |
| 4 | Testing Contract |
| 5 | FR12.2, FR12.3 |
| 6 | Testing Contract |
| 7 | FR12.1, FR12.3, NFR4.1, NFR5.1, NFR2.1 |
| 8 | Testing Contract |
| 9 | Contrato 5, FR12 |
