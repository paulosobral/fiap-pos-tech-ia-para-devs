# Code Generation Plan — u5-anomaly

> Unidade U5 (Anomaly — Detecção de Anomalias). Escopo: `feature` · Estratégia: `standard` · Metodologia: `test-after` (Testing Contract do time). Greenfield.
> Fontes: contract-summary.md (Contratos 5/6/7), unit-of-work.md, requirements.md (FR9.1–FR9.4, NFR5.1, NFR2.1, NFR4.1), referência de convenções: u4-async-ingest (estrutura e test-after), u1-core-conversation (schema de sessões/conversas), u2/u3 (logs estruturados e wiring por DI).

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

- [x] **Step 1** — Estrutura do projeto e configuração de produção (layout `apps/anomaly-detector/{handler.py, service/, infra/, tests/, requirements.txt}` conforme PRD §7.4). *(História: setup da unidade U5)*
- [x] **Step 2** — Bootstrap do runner de testes (pytest + pytest-cov já no `.venv` da raiz; path bootstrap via `tests/conftest.py`, pyproject da raiz intocado) e registro do comando exato com escopo da unidade. *(Testing Contract: runner antes do primeiro teste)*
- [x] **Step 3** — Camada de dados: `infra/conversation_store.py` ConversationStore (leitor do Contrato 6 — scan paginado com filtro `begins_with(SK, CONV#)`, unmarshal de tipos e campos JSON, itens sem identificadores ignorados com log) e `infra/alert_store.py` AlertStore (owner do Contrato 7: put de anomalias com `PK=anomaly_id`, update de restrição de agendamento, consulta de restrição aberta por lead via GSI `lead-index`). *(FR9.3, Contratos 5/6/7)*
- [x] **Step 4** — Testes das camadas de dados (test-after). *(Testing Contract)*
- [x] **Step 5** — Features e scoring: `service/feature_extractor.py` ConversationFeatureExtractor (FR9.1 — volume, comprimento médio, sentimento heurístico pt-BR, horários atípicos; função pura e determinística) e `service/scorer.py` (Protocol `AnomalyScorer` + `HeuristicScorer` determinístico 100% stdlib + `SklearnScorer` Isolation Forest + PCA com erro de reconstrução como sinal residual estilo autoencoder, imports guardados e fallback por lote pequeno). *(FR9.1, FR9.2)*
- [x] **Step 6** — Testes de features e scorer (test-after). *(Testing Contract)*
- [x] **Step 7** — Gate + orquestrador + handler: `service/scheduler_gate.py` SchedulingGate (restrição de agendamento materializada no item da anomalia e consultável por lead), `service/anomaly_detector.py` AnomalyDetector (pipeline diário: ler conversas → features → score → alerta Contrato 7 → restringir agendamento; `anomaly_id` determinístico `session_id#data` idempotente; logs estruturados PII-safe; clock injetável) e `handler.py` (wiring EventBridge com boto3/env, `ANOMALY_SCORER`/`ANOMALY_THRESHOLD` configuráveis). *(FR9.1, FR9.3, FR9.4, NFR5.1, NFR2.1, NFR4.1)*
- [x] **Step 8** — Testes do orquestrador + integração (pipeline ponta a ponta com stores fake, idempotência do mesmo dia, restrição consultável por lead, PII-safe nos logs, wiring do `handler()` com EventBridge, degradação do sklearn ausente). *(Testing Contract)*
- [x] **Step 9** — Configuração de build/deploy (`requirements.txt`) e documentação/traceability (`source-manifest.json`, `traceability.json`). *(Contrato 7, FR9)*

## Rastreabilidade passo → requisito

| Step | Requisito(s) |
|------|--------------|
| 1 | — (estrutura) |
| 2 | Testing Contract |
| 3 | FR9.3, Contratos 5/6/7 |
| 4 | Testing Contract |
| 5 | FR9.1, FR9.2 |
| 6 | Testing Contract |
| 7 | FR9.1, FR9.3, FR9.4, NFR5.1, NFR2.1, NFR4.1 |
| 8 | Testing Contract |
| 9 | Contrato 7, FR9 |
