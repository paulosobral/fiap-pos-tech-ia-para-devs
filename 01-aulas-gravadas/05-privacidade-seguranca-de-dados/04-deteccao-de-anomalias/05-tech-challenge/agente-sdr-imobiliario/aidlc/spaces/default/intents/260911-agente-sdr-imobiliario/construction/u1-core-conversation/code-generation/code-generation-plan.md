# Code Generation Plan — u1-core-conversation

> Unidade U1 (Core Conversation). Escopo: `feature` · Estratégia: `standard` · Metodologia: `test-after` (Testing Contract do time). Greenfield.
> Fontes: functional-spec.md, rules.md, entities.md, contract-summary.md, unit-of-work.md, requirements.md.

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

- [x] **Step 1** — Estrutura do projeto e configuração de produção (pyproject/requirements, layout `apps/conversation-router/{handler.py, service/, infra/, tests/}` conforme PRD §7.4). *(História: setup da unidade U1)*
- [x] **Step 2** — Bootstrap do runner de testes (pytest + pytest-cov) e registro do comando exato com escopo da unidade. *(Testing Contract: runner antes do primeiro teste)*
- [x] **Step 3** — Camada de dados: entidades Lead/Conversation e persistência DynamoDB. *(FR1.3, R8, entities.md)*
- [x] **Step 4** — Testes da camada de dados (test-after). *(Testing Contract)*
- [x] **Step 5** — Lógica de negócio: SecurityLayer (PII masking R1, guardrails R7), LeadQualifier (score R3), SalesFlow (LangGraph: greeting/elicitation/intent/qualification/recommendation/scheduling/handoff/followup, R2/R4/R5/R6). *(FR1-FR6, FR10)*
- [x] **Step 6** — Testes da lógica de negócio (test-after). *(Testing Contract)*
- [x] **Step 7** — API/endpoint: handler Lambda `conversation-router` (webhook Telegram: 200/401/400), enfileiramento SQS (voice, CRM), sessão DynamoDB. *(Contrato POST /webhook, R1, R8)*
- [x] **Step 8** — Testes do endpoint (integração: handler com mocks). *(Testing Contract)*
- [x] **Step 9** — Configuração de build/deploy (empacotamento Lambda) e documentação/traceability. *(NFR9, contratos SQS)*

## Rastreabilidade passo → requisito

| Step | Requisito(s) |
|------|--------------|
| 1 | — (estrutura) |
| 2 | Testing Contract |
| 3 | FR1.3, R8, NFR2.5 |
| 5 | FR1.1, FR2.1-2.3, FR3.1-3.3, FR4.2-4.3, FR5.1-5.4, FR6.1-6.3, FR10.1-10.3, R1-R7 |
| 7 | FR1.1, NFR2.1, NFR4.2, R1, R8 |
