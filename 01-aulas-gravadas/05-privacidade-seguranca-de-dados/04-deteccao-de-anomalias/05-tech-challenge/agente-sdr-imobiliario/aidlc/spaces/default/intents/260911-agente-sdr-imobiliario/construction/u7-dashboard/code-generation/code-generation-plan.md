# Code Generation Plan — u7-dashboard

> Unidade U7 (Dashboard — DashAPI + Dashboard Streamlit Community Cloud). Escopo: `feature` · Estratégia: `standard` · Metodologia: `test-after` (Testing Contract do time). Greenfield.
> Fontes: contract-summary.md (Contratos 2/5/7), unit-of-work.md (U7), requirements.md (FR7.1–FR7.4, FR9.3, FR10.1, FR10.2, NFR5.1, NFR2.1, NFR4.1), referência de convenções: u5-anomaly (schema dono da tabela de alertas), u6-followup (leitor do Contrato 5 com dois scans, wiring por DI, test-after), u4-async-ingest (SessionStore/reader pattern), u1-core-conversation (estados da esteira `VALID_STATES`, regra da roleta `AREA_THRESHOLD_M2 = 500`).

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

Obrigações: 5-8 testes por componente; testes de unidade + integração para boundaries chave; piso de cobertura de 80% de linhas sobre `apps/dashboard-api` (a UI carrega apenas smoke test); execução em CI antes do merge. Nenhuma meta pode ser relaxada para fazer uma etapa passar.

## Steps

- [x] **Step 1** — Estrutura do projeto e configuração de produção (layout `apps/dashboard-api/{handler.py, service/, infra/, tests/, requirements.txt}` e `apps/dashboard-ui/{app.py, tests/, requirements.txt}` conforme PRD §7.4). *(História: setup da unidade U7)*
- [x] **Step 2** — Bootstrap do runner de testes (pytest + pytest-cov já no `.venv` da raiz; path bootstrap via `tests/conftest.py` em cada app, pyproject da raiz intocado; suíte única cobrindo os dois diretórios — sem `__init__.py` nos `tests/` para evitar colisão de módulo `tests` no pytest) e registro do comando exato com escopo da unidade. *(Testing Contract: runner antes do primeiro teste)*
- [x] **Step 3** — Camada de dados: `infra/conversation_store.py` ConversationStore (leitor do Contrato 5 — scan paginado `begins_with(SK, CONV#)` para conversas e `begins_with(SK, PROFILE)` para o mapa `lead_id → perfil`, unmarshal de tipos e campos JSON, itens sem identificadores ignorados com log, erros → `ConversationStoreError`), `infra/alert_store.py` AlertStoreReader (leitor do Contrato 7 — schema dono U5: `anomaly_id`, `lead_id`, `features`, `confidence`, `type`, `detected_at`, `status`, `action_taken`; ordena por `detected_at` desc; `project_alert` PII-safe) e `infra/metrics.py` CloudWatchMeter (métricas `response_time_p90`/`cost_monthly` via `get_metric_data` com cliente injetado; falta de datapoint/erro → `None` com degradação graciosa). *(FR7.1, FR7.4, Contrato 2, Contrato 5, Contrato 7)*
- [x] **Step 4** — Testes das camadas de dados (test-after). *(Testing Contract)*
- [x] **Step 5** — Cálculo e agregação: `service/kpi_calc.py` funções puras determinísticas (`parse_iso8601`, `count_new_leads`, `latest_conversation_by_lead`, `qualification_rate`, `state_funnel` com os 8 estados da esteira do u1 + bucket `outros`, `scheduled_visits_count`, `intent_volume`, `route_distribution` regra da roleta do u1 — ≤ 500 m² → rodízio/consultores, > 500 m² → diretor/especialista, com override por campo `route`/`route_target`/`assigned_to`, `alerts_last_24h`) e `service/kpis.py` KpiService (orquestra conversas + perfis + alertas + meter → snapshot do Contrato 2 com campos contratuais + extensões aditivas `generated_at`/`intents`/`route_distribution`/`funnel`/`alerts`; falha de store propaga → 500, tabela vazia/item inválido → zeros/estado vazio; logs estruturados `log_event`). *(FR7.2, FR7.3, FR7.4, Contrato 2)*
- [x] **Step 6** — Testes de cálculo e agregação (test-after). *(Testing Contract)*
- [x] **Step 7** — Handler + UI: `handler.py` (Lambda aws_proxy `GET /api/kpis` do Contrato 2 — 200 JSON, 401 sem bearer `Authorization` (CognitoAuthorizer na borda do API Gateway em produção), 404 path, 405 método, 500 em falha de agregação; wiring DI via `build_service` com boto3 preguiçoso; header CORS via `DASHBOARD_ALLOWED_ORIGIN`) e `apps/dashboard-ui/app.py` (1 página Streamlit — `fetch_kpis` com http injetado e bearer de `DASHBOARD_API_TOKEN`, cards FR7.2, esteira Kanban + gráficos de intenções e roleta FR7.3, tabela de alertas 24h FR7.4, toast de erro 500, callout de login Cognito via `COGNITO_LOGIN_URL`; sem persistência local). *(FR7.1–FR7.4, Contrato 2, NFR5.1, NFR2.1, NFR4.1, RT6)*
- [x] **Step 8** — Testes do handler + integração ponta a ponta (fake boto3 em `sys.modules` + Dynamo fake com filtro de prefixo; 401/404/405/500; CORS; zeros em tabelas vazias; itens malformados ignorados; PII-safe no corpo e nos logs; CloudWatch fora do ar → zeros com 200) + smoke da UI com `importorskip("streamlit")`. *(Testing Contract)*
- [x] **Step 9** — Configuração de build/deploy (`apps/dashboard-api/requirements.txt` = boto3; `apps/dashboard-ui/requirements.txt` = streamlit + requests) e documentação/traceability (`source-manifest.json`, `traceability.json`). *(Contrato 2, FR7)*

## Rastreabilidade passo → requisito

| Step | Requisito(s) |
|------|--------------|
| 1 | — (estrutura) |
| 2 | Testing Contract |
| 3 | FR7.1, FR7.4, Contrato 2, Contrato 5, Contrato 7 |
| 4 | Testing Contract |
| 5 | FR7.2, FR7.3, FR7.4, Contrato 2, FR10.1, FR10.2 |
| 6 | Testing Contract |
| 7 | FR7.1, FR7.2, FR7.3, FR7.4, Contrato 2, NFR5.1, NFR2.1, NFR4.1, RT6 |
| 8 | Testing Contract |
| 9 | Contrato 2, FR7 |
